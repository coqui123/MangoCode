//! Voice input: availability checks, hold-to-talk recording, and speech-to-text
//! transcription via a local whisper.cpp-compatible command.
//!
//! # Feature flag
//! Audio capture via `cpal` is gated behind the `voice` feature.  When the
//! feature is disabled the recorder still compiles but `start_recording` returns
//! an error immediately rather than attempting hardware access.

use crate::oauth::OAuthTokens;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use tokio::sync::mpsc;

// ---------------------------------------------------------------------------
// Availability (kill-switch / compatibility)
// ---------------------------------------------------------------------------

/// Environment variable that disables voice mode when set (any value)
const KILL_SWITCH_ENV: &str = "MANGOCODE_VOICE_DISABLED";

/// Whether voice mode is available.
#[derive(Debug, Clone, PartialEq)]
pub enum VoiceAvailability {
    Available,
    /// Deprecated: local voice no longer requires OAuth.
    RequiresOAuth,
    /// Deprecated: local voice no longer requires OAuth scopes.
    MissingScopes {
        required: Vec<String>,
        have: Vec<String>,
    },
    /// Feature disabled by kill-switch environment variable
    Disabled,
    /// Feature flag not enabled in this build
    NotEnabled,
    /// No microphone / audio device available on this system
    NoMicrophone {
        reason: String,
    },
    /// Voice input is enabled but the user has toggled it off
    ToggledOff,
}

impl VoiceAvailability {
    /// Returns `true` when voice mode can be started.
    pub fn is_available(&self) -> bool {
        matches!(self, VoiceAvailability::Available)
    }

    /// Returns a human-readable error message when voice is not available,
    /// or `None` when it is.
    pub fn error_message(&self) -> Option<String> {
        match self {
            VoiceAvailability::Available => None,
            VoiceAvailability::RequiresOAuth => Some(
                "Voice mode no longer requires OAuth. Check local voice configuration.".to_string(),
            ),
            VoiceAvailability::MissingScopes { required, have } => Some(format!(
                "Voice mode no longer requires OAuth scopes. Legacy required scopes: {}. Your token has: {}",
                required.join(", "),
                if have.is_empty() {
                    "none".to_string()
                } else {
                    have.join(", ")
                }
            )),
            VoiceAvailability::Disabled => Some("Voice mode is currently disabled.".to_string()),
            VoiceAvailability::NotEnabled => {
                Some("Voice mode is not enabled in this build.".to_string())
            }
            VoiceAvailability::NoMicrophone { reason } => Some(reason.clone()),
            VoiceAvailability::ToggledOff => {
                Some("Voice input is disabled. Run /voice to enable.".to_string())
            }
        }
    }
}

/// Check whether voice mode is available.
///
/// The `tokens` argument is retained for compatibility; local voice
/// transcription no longer requires OAuth or API-key authentication.
pub fn check_voice_availability(_tokens: Option<&OAuthTokens>) -> VoiceAvailability {
    // Check kill switch first — always wins
    if std::env::var(KILL_SWITCH_ENV).is_ok() {
        return VoiceAvailability::Disabled;
    }

    VoiceAvailability::Available
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the voice recorder / transcription pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceConfig {
    /// Whether the user has enabled voice input.
    pub enabled: bool,
    /// Deprecated: voice transcription is local-only and does not use API keys.
    /// Kept for backwards-compatible config deserialization.
    pub api_key: Option<String>,
    /// BCP-47 language hint sent to the local transcription command (e.g. `"en"`).
    /// When `None` whisper.cpp auto-detects the language.
    pub language: Option<String>,
    /// Backwards-compatible model field. If this is set to a filesystem path,
    /// it is used as the local Whisper model path.
    pub model: String,
    /// Deprecated: voice transcription is local-only and does not use endpoints.
    /// Kept for backwards-compatible config deserialization.
    pub endpoint_url: Option<String>,
    /// Path to a local whisper.cpp GGML model, such as ggml-tiny.en.bin.
    /// Can also be supplied with MANGOCODE_WHISPER_MODEL.
    pub model_path: Option<String>,
    /// Path to a local whisper.cpp-compatible executable.
    /// Defaults to MANGOCODE_WHISPER_BIN, then whisper-cli/whisper/main on PATH.
    pub local_command: Option<String>,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            api_key: None,
            language: None,
            model: "whisper-1".to_string(),
            endpoint_url: None,
            model_path: None,
            local_command: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// Events produced by the voice recorder.
#[derive(Debug, Clone)]
pub enum VoiceEvent {
    /// Recording has begun; the UI should show a recording indicator.
    RecordingStarted,
    /// Recording has stopped; transcription is in progress.
    RecordingStopped,
    /// Transcription succeeded; the contained string should be inserted into
    /// the input box.
    TranscriptReady(String),
    /// An error occurred.  The string is a human-readable message.
    Error(String),
}

// ---------------------------------------------------------------------------
// Recorder
// ---------------------------------------------------------------------------

/// Hold-to-talk voice recorder that captures microphone audio and sends it to
/// a local whisper.cpp-compatible speech-to-text command.
pub struct VoiceRecorder {
    is_enabled: bool,
    is_recording: Arc<AtomicBool>,
    config: VoiceConfig,
}

impl VoiceRecorder {
    /// Create a new recorder from the given configuration.
    pub fn new(config: VoiceConfig) -> Self {
        let is_enabled = config.enabled;
        Self {
            is_enabled,
            is_recording: Arc::new(AtomicBool::new(false)),
            config,
        }
    }

    /// Check if voice input is available on this system.
    pub fn check_availability(&mut self) -> VoiceAvailability {
        if !self.is_enabled {
            return VoiceAvailability::ToggledOff;
        }
        check_microphone_availability()
    }

    /// Enable or disable voice input.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.is_enabled = enabled;
        self.config.enabled = enabled;
    }

    /// Returns `true` while audio is being captured.
    pub fn is_recording(&self) -> bool {
        self.is_recording.load(Ordering::SeqCst)
    }

    /// Begin recording audio.  Voice events are delivered over `event_tx`.
    ///
    /// This is a non-blocking call: audio capture and transcription run on
    /// Tokio tasks that stay alive until `stop_recording` is called (or the
    /// recorder is dropped).
    pub async fn start_recording(
        &mut self,
        event_tx: mpsc::Sender<VoiceEvent>,
    ) -> anyhow::Result<()> {
        if self.is_recording.load(Ordering::SeqCst) {
            return Ok(());
        }

        let availability = self.check_availability();
        if !availability.is_available() {
            let msg = availability
                .error_message()
                .unwrap_or_else(|| "Voice unavailable".to_string());
            let _ = event_tx.send(VoiceEvent::Error(msg.clone())).await;
            return Err(anyhow::anyhow!(msg));
        }

        self.is_recording.store(true, Ordering::SeqCst);

        let is_recording = self.is_recording.clone();
        let config = self.config.clone();

        // cpal::Stream is !Send, so we can't use tokio::spawn (which requires Send).
        // Instead, spin up a dedicated OS thread with its own single-threaded tokio
        // runtime so the stream stays local to that thread throughout its lifetime.
        std::thread::spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(err) => {
                    let _ = event_tx.blocking_send(VoiceEvent::Error(format!(
                        "failed to initialize voice runtime: {}",
                        err
                    )));
                    return;
                }
            };
            rt.block_on(async move {
                match record_and_transcribe(is_recording, event_tx.clone(), config).await {
                    Ok(()) => {}
                    Err(e) => {
                        let _ = event_tx.send(VoiceEvent::Error(e.to_string())).await;
                    }
                }
            });
        });

        Ok(())
    }

    /// Stop recording.  The transcription request is sent immediately after
    /// the audio capture loop exits.
    pub async fn stop_recording(&mut self) -> anyhow::Result<()> {
        self.is_recording.store(false, Ordering::SeqCst);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Microphone availability check (platform-aware)
// ---------------------------------------------------------------------------

fn check_microphone_availability() -> VoiceAvailability {
    #[cfg(feature = "voice")]
    {
        use cpal::traits::HostTrait;
        let host = cpal::default_host();
        if host.default_input_device().is_none() {
            return VoiceAvailability::NoMicrophone {
                reason: platform_no_mic_message(),
            };
        }
        VoiceAvailability::Available
    }
    #[cfg(not(feature = "voice"))]
    {
        VoiceAvailability::NotEnabled
    }
}

#[cfg(feature = "voice")]
fn platform_no_mic_message() -> String {
    if cfg!(target_os = "windows") {
        "No microphone found. Go to Settings \u{2192} Privacy \u{2192} Microphone to grant access, then connect a microphone.".to_string()
    } else if cfg!(target_os = "macos") {
        "No microphone found. Check System Settings \u{2192} Privacy & Security \u{2192} Microphone.".to_string()
    } else {
        "No microphone found. Connect a microphone and ensure your audio system is configured correctly.".to_string()
    }
}

// ---------------------------------------------------------------------------
// Recording + transcription pipeline
// ---------------------------------------------------------------------------

/// Captures audio while `is_recording` is `true`, then transcribes and sends
/// the result over `event_tx`.
async fn record_and_transcribe(
    is_recording: Arc<AtomicBool>,
    event_tx: mpsc::Sender<VoiceEvent>,
    config: VoiceConfig,
) -> anyhow::Result<()> {
    #[cfg(feature = "voice")]
    {
        let (samples, sample_rate) = record_audio(is_recording, event_tx.clone()).await?;

        let _ = event_tx.send(VoiceEvent::RecordingStopped).await;

        if samples.is_empty() {
            return Ok(());
        }

        match transcribe_audio(&samples, sample_rate, &config).await {
            Ok(text) => {
                let _ = event_tx.send(VoiceEvent::TranscriptReady(text)).await;
            }
            Err(e) => {
                let _ = event_tx
                    .send(VoiceEvent::Error(format!("Transcription failed: {}", e)))
                    .await;
            }
        }
        Ok(())
    }
    #[cfg(not(feature = "voice"))]
    {
        let _ = is_recording;
        let _ = config;
        let msg = "Voice recording is not available in this build (compile with --features voice)."
            .to_string();
        let _ = event_tx.send(VoiceEvent::Error(msg.clone())).await;
        Err(anyhow::anyhow!(msg))
    }
}

// ---------------------------------------------------------------------------
// Audio capture (cpal, feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "voice")]
async fn record_audio(
    is_recording: Arc<AtomicBool>,
    event_tx: mpsc::Sender<VoiceEvent>,
) -> anyhow::Result<(Vec<f32>, u32)> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::SampleFormat;
    use std::time::Duration;

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device available"))?;

    let supported_config = device.default_input_config()?;
    let sample_rate = supported_config.sample_rate().0;
    let channels = supported_config.channels() as usize;

    let samples: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let samples_clone = samples.clone();

    let sample_format = supported_config.sample_format();
    let config: cpal::StreamConfig = supported_config.into();
    let stream = match sample_format {
        SampleFormat::F32 => build_input_stream(&device, &config, channels, samples_clone, |s| s)?,
        SampleFormat::F64 => {
            build_input_stream(&device, &config, channels, samples_clone, |s: f64| s as f32)?
        }
        SampleFormat::I8 => {
            build_input_stream(&device, &config, channels, samples_clone, |s: i8| {
                s as f32 / i8::MAX as f32
            })?
        }
        SampleFormat::I16 => {
            build_input_stream(&device, &config, channels, samples_clone, |s: i16| {
                s as f32 / i16::MAX as f32
            })?
        }
        SampleFormat::I32 => {
            build_input_stream(&device, &config, channels, samples_clone, |s: i32| {
                s as f32 / i32::MAX as f32
            })?
        }
        SampleFormat::U8 => {
            build_input_stream(&device, &config, channels, samples_clone, |s: u8| {
                (s as f32 - 128.0) / 128.0
            })?
        }
        SampleFormat::U16 => {
            build_input_stream(&device, &config, channels, samples_clone, |s: u16| {
                (s as f32 - 32768.0) / 32768.0
            })?
        }
        SampleFormat::U32 => {
            build_input_stream(&device, &config, channels, samples_clone, |s: u32| {
                (s as f32 - 2147483648.0) / 2147483648.0
            })?
        }
        other => {
            return Err(anyhow::anyhow!(
                "Unsupported microphone sample format: {:?}",
                other
            ));
        }
    };

    stream.play()?;
    let _ = event_tx.send(VoiceEvent::RecordingStarted).await;

    while is_recording.load(Ordering::SeqCst) {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    drop(stream);
    let audio = samples
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    Ok((audio, sample_rate))
}

#[cfg(feature = "voice")]
fn build_input_stream<T, F>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    samples: Arc<Mutex<Vec<f32>>>,
    convert: F,
) -> anyhow::Result<cpal::Stream>
where
    T: cpal::SizedSample,
    F: Fn(T) -> f32 + Send + Sync + Copy + 'static,
{
    use cpal::traits::DeviceTrait;

    Ok(device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            let mut s = samples
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if channels == 1 {
                s.extend(data.iter().copied().map(convert));
            } else {
                for chunk in data.chunks(channels) {
                    let mono = chunk.iter().copied().map(convert).sum::<f32>() / channels as f32;
                    s.push(mono);
                }
            }
        },
        move |err| {
            tracing::error!("Audio stream error: {}", err);
        },
        None,
    )?)
}

// ---------------------------------------------------------------------------
// WAV encoding
// ---------------------------------------------------------------------------

/// Encode mono 32-bit float PCM samples as a standard WAV file (16-bit PCM).
#[cfg_attr(not(feature = "voice"), allow(dead_code))]
fn encode_wav(samples: &[f32], sample_rate: u32) -> anyhow::Result<Vec<u8>> {
    let num_samples = samples.len() as u32;
    let byte_rate = sample_rate * 2; // 16-bit mono → 2 bytes/sample
    let data_size = num_samples * 2;
    let total_size = 44 + data_size;

    let mut buf = Vec::with_capacity(total_size as usize);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(total_size - 8).to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt  chunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align (1 ch × 2 bytes)
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    for &sample in samples {
        let s = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        buf.extend_from_slice(&s.to_le_bytes());
    }

    Ok(buf)
}

// ---------------------------------------------------------------------------
// Speech-to-text transcription
// ---------------------------------------------------------------------------

/// Transcribe `audio_samples` locally with whisper.cpp and return the transcript
/// text. No network requests are made.
#[cfg_attr(not(feature = "voice"), allow(dead_code))]
async fn transcribe_audio(
    audio_samples: &[f32],
    sample_rate: u32,
    config: &VoiceConfig,
) -> anyhow::Result<String> {
    #[cfg(feature = "voice")]
    {
        let model_path = resolve_whisper_model_path(config)?;
        let command = resolve_whisper_command(config)?;
        let language = config.language.clone();
        let samples = resample_to_16khz(audio_samples, sample_rate);

        tokio::task::spawn_blocking(move || {
            transcribe_audio_local(&samples, &model_path, &command, language.as_deref())
        })
        .await?
    }

    #[cfg(not(feature = "voice"))]
    {
        let _ = audio_samples;
        let _ = sample_rate;
        let _ = config;
        Err(anyhow::anyhow!(
            "Local voice transcription is not available in this build (compile with --features voice)."
        ))
    }
}

#[cfg(feature = "voice")]
fn transcribe_audio_local(
    samples_16khz: &[f32],
    model_path: &Path,
    command: &Path,
    language: Option<&str>,
) -> anyhow::Result<String> {
    let wav_path =
        std::env::temp_dir().join(format!("mangocode-voice-{}.wav", uuid::Uuid::new_v4()));
    let output_base =
        std::env::temp_dir().join(format!("mangocode-voice-{}", uuid::Uuid::new_v4()));
    let output_txt = output_base.with_extension("txt");

    std::fs::write(&wav_path, encode_wav(samples_16khz, 16_000)?)?;

    let mut cmd = Command::new(command);
    cmd.arg("-m")
        .arg(model_path)
        .arg("-f")
        .arg(&wav_path)
        .arg("-otxt")
        .arg("-of")
        .arg(&output_base)
        .arg("-nt")
        .arg("-np");

    if let Some(language) = language.filter(|s| !s.trim().is_empty()) {
        cmd.arg("-l").arg(language);
    }

    let output = cmd.output();
    let _ = std::fs::remove_file(&wav_path);

    let output = output?;
    if !output.status.success() {
        let _ = std::fs::remove_file(&output_txt);
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!(
            "Local transcription command failed: {}",
            stderr.trim()
        ));
    }

    let text = match std::fs::read_to_string(&output_txt) {
        Ok(text) => {
            let _ = std::fs::remove_file(&output_txt);
            text
        }
        Err(_) => String::from_utf8_lossy(&output.stdout).to_string(),
    };

    Ok(clean_whisper_output(&text))
}

#[cfg(feature = "voice")]
fn resolve_whisper_model_path(config: &VoiceConfig) -> anyhow::Result<PathBuf> {
    let configured = config
        .model_path
        .as_deref()
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("MANGOCODE_WHISPER_MODEL")
                .ok()
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
                .map(PathBuf::from)
        })
        .or_else(|| {
            let model = config.model.trim();
            if model != "whisper-1"
                && (model.ends_with(".bin") || model.contains('/') || model.contains('\\'))
            {
                Some(PathBuf::from(model))
            } else {
                None
            }
        });

    let Some(path) = configured else {
        return Err(anyhow::anyhow!(
            "Local voice transcription requires a whisper.cpp GGML model file. \
             Set MANGOCODE_WHISPER_MODEL to a local model path, for example ggml-tiny.en.bin."
        ));
    };

    if !path.is_file() {
        return Err(anyhow::anyhow!(
            "Whisper model file not found: {}",
            path.display()
        ));
    }

    Ok(path)
}

#[cfg(feature = "voice")]
fn resolve_whisper_command(config: &VoiceConfig) -> anyhow::Result<PathBuf> {
    let configured = config
        .local_command
        .as_deref()
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("MANGOCODE_WHISPER_BIN")
                .ok()
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
                .map(PathBuf::from)
        });

    if let Some(path) = configured {
        return Ok(path);
    }

    for candidate in ["whisper-cli", "whisper", "main"] {
        if command_exists(candidate) {
            return Ok(PathBuf::from(candidate));
        }
    }

    Err(anyhow::anyhow!(
        "Local voice transcription requires whisper.cpp on PATH. \
         Install whisper.cpp and set MANGOCODE_WHISPER_BIN to whisper-cli.exe."
    ))
}

#[cfg(feature = "voice")]
fn command_exists(command: &str) -> bool {
    Command::new(command)
        .arg("--help")
        .output()
        .map(|output| {
            output.status.success() || !output.stderr.is_empty() || !output.stdout.is_empty()
        })
        .unwrap_or(false)
}

#[cfg(feature = "voice")]
fn resample_to_16khz(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    const TARGET_RATE: u32 = 16_000;
    if sample_rate == TARGET_RATE || samples.is_empty() {
        return samples.to_vec();
    }

    let output_len = ((samples.len() as u64 * TARGET_RATE as u64) / sample_rate as u64) as usize;
    let mut out = Vec::with_capacity(output_len);
    let ratio = sample_rate as f64 / TARGET_RATE as f64;

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos.floor() as usize;
        let frac = (src_pos - idx as f64) as f32;
        let a = samples.get(idx).copied().unwrap_or(0.0);
        let b = samples.get(idx + 1).copied().unwrap_or(a);
        out.push(a + (b - a) * frac);
    }

    out
}

#[cfg(feature = "voice")]
fn clean_whisper_output(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| !line.starts_with("whisper_"))
        .filter(|line| !line.starts_with("system_info:"))
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------

use once_cell::sync::Lazy;

static GLOBAL_VOICE_RECORDER: Lazy<Arc<Mutex<VoiceRecorder>>> =
    Lazy::new(|| Arc::new(Mutex::new(VoiceRecorder::new(VoiceConfig::default()))));

/// Access the global `VoiceRecorder` instance.
///
/// Callers should call `set_enabled(true)` and update the config before
/// invoking `start_recording`.
pub fn global_voice_recorder() -> Arc<Mutex<VoiceRecorder>> {
    GLOBAL_VOICE_RECORDER.clone()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Serialize all tests that read or write `KILL_SWITCH_ENV` so they don't
    /// interfere with each other when the test runner runs them in parallel.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn tokens_with_scopes(scopes: Vec<&str>) -> OAuthTokens {
        OAuthTokens {
            access_token: "test_token".to_string(),
            scopes: scopes.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn test_no_tokens_available_for_local_voice() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var(KILL_SWITCH_ENV);
        let result = check_voice_availability(None);
        assert_eq!(result, VoiceAvailability::Available);
        assert!(result.is_available());
        assert!(result.error_message().is_none());
    }

    #[test]
    fn test_available_with_all_scopes() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var(KILL_SWITCH_ENV);
        let tokens = tokens_with_scopes(vec!["user:inference", "user:profile"]);
        let result = check_voice_availability(Some(&tokens));
        assert_eq!(result, VoiceAvailability::Available);
        assert!(result.is_available());
        assert!(result.error_message().is_none());
    }

    #[test]
    fn test_missing_one_scope_still_available_for_local_voice() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var(KILL_SWITCH_ENV);
        let tokens = tokens_with_scopes(vec!["user:inference"]);
        let result = check_voice_availability(Some(&tokens));
        assert_eq!(result, VoiceAvailability::Available);
        assert!(result.is_available());
    }

    #[test]
    fn test_missing_all_scopes_still_available_for_local_voice() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var(KILL_SWITCH_ENV);
        let tokens = tokens_with_scopes(vec!["org:create_api_key"]);
        let result = check_voice_availability(Some(&tokens));
        assert_eq!(result, VoiceAvailability::Available);
        assert!(result.is_available());
    }

    #[test]
    fn test_empty_scopes_still_available_for_local_voice() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var(KILL_SWITCH_ENV);
        let tokens = tokens_with_scopes(vec![]);
        let result = check_voice_availability(Some(&tokens));
        assert_eq!(result, VoiceAvailability::Available);
        assert!(result.is_available());
    }

    #[test]
    fn test_kill_switch_disables_voice() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var(KILL_SWITCH_ENV, "1");
        let tokens = tokens_with_scopes(vec!["user:inference", "user:profile"]);
        let result = check_voice_availability(Some(&tokens));
        std::env::remove_var(KILL_SWITCH_ENV);
        assert_eq!(result, VoiceAvailability::Disabled);
        assert!(!result.is_available());
    }

    #[test]
    fn test_kill_switch_beats_no_auth() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var(KILL_SWITCH_ENV, "true");
        let result = check_voice_availability(None);
        std::env::remove_var(KILL_SWITCH_ENV);
        assert_eq!(result, VoiceAvailability::Disabled);
    }

    #[test]
    fn test_not_enabled_error_message() {
        let v = VoiceAvailability::NotEnabled;
        assert!(!v.is_available());
        assert!(v.error_message().unwrap().contains("not enabled"));
    }

    #[test]
    fn test_extra_scopes_still_available() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var(KILL_SWITCH_ENV);
        let tokens = tokens_with_scopes(vec![
            "user:inference",
            "user:profile",
            "org:create_api_key",
            "user:file_upload",
        ]);
        let result = check_voice_availability(Some(&tokens));
        assert_eq!(result, VoiceAvailability::Available);
    }

    #[test]
    fn test_voice_config_default() {
        let cfg = VoiceConfig::default();
        assert!(!cfg.enabled);
        assert!(cfg.api_key.is_none());
        assert_eq!(cfg.model, "whisper-1");
    }

    #[test]
    fn test_recorder_not_recording_initially() {
        let rec = VoiceRecorder::new(VoiceConfig::default());
        assert!(!rec.is_recording());
    }

    #[test]
    fn test_encode_wav_produces_valid_header() {
        let samples: Vec<f32> = vec![0.0f32; 16];
        let wav = encode_wav(&samples, 16000).unwrap();
        // RIFF magic
        assert_eq!(&wav[0..4], b"RIFF");
        // WAVE magic
        assert_eq!(&wav[8..12], b"WAVE");
        // fmt  chunk id
        assert_eq!(&wav[12..16], b"fmt ");
        // data chunk id
        assert_eq!(&wav[36..40], b"data");
        // Total: 44 (header) + 16*2 (samples) = 76
        assert_eq!(wav.len(), 76);
    }

    #[test]
    fn test_encode_wav_clamps_samples() {
        let samples = vec![2.0f32, -2.0f32];
        let wav = encode_wav(&samples, 44100).unwrap();
        // 44 byte header + 4 bytes data
        assert_eq!(wav.len(), 48);
        // First sample should be i16::MAX (32767)
        let s0 = i16::from_le_bytes([wav[44], wav[45]]);
        assert_eq!(s0, 32767);
        // Second sample should be i16::MIN-equivalent (-32767)
        let s1 = i16::from_le_bytes([wav[46], wav[47]]);
        assert_eq!(s1, -32767);
    }

    #[test]
    fn test_toggled_off_message() {
        let v = VoiceAvailability::ToggledOff;
        assert!(!v.is_available());
        assert!(v.error_message().unwrap().contains("/voice"));
    }

    #[test]
    fn test_no_microphone_message() {
        let v = VoiceAvailability::NoMicrophone {
            reason: "No mic".to_string(),
        };
        assert!(!v.is_available());
        assert_eq!(v.error_message().unwrap(), "No mic");
    }

    #[test]
    fn test_global_voice_recorder_is_consistent() {
        let r1 = global_voice_recorder();
        let r2 = global_voice_recorder();
        // Both arcs point to the same allocation
        assert!(Arc::ptr_eq(&r1, &r2));
    }

    #[test]
    fn test_set_enabled() {
        let mut rec = VoiceRecorder::new(VoiceConfig::default());
        assert!(!rec.config.enabled);
        rec.set_enabled(true);
        assert!(rec.config.enabled);
        rec.set_enabled(false);
        assert!(!rec.config.enabled);
    }

    #[test]
    fn test_voice_config_serialization() {
        let cfg = VoiceConfig {
            enabled: true,
            api_key: Some("sk-test".to_string()),
            language: Some("en".to_string()),
            model: "whisper-1".to_string(),
            endpoint_url: None,
            model_path: None,
            local_command: None,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: VoiceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "whisper-1");
        assert_eq!(back.api_key.as_deref(), Some("sk-test"));
    }
}
