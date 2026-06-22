// ConfigTool: get or set MangoCode configuration settings at runtime.
//
// Reads from and persists to ~/.mangocode/settings.json.
// Supported settings: model, max_tokens, verbose, permission_mode, reliability controls.

use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use mangocode_core::config::{
    AgentCompletionPolicy, AgentReliabilityProfile, AgentSpeedProfile, Config, Settings,
    VerificationPolicy,
};
use serde::Deserialize;
use serde_json::{json, Value};

pub struct ConfigTool;

#[derive(Debug, Deserialize)]
struct ConfigInput {
    setting: String,
    value: Option<Value>,
}

static SUPPORTED_SETTINGS: &[(&str, &str)] = &[
    (
        "model",
        "LLM model to use (e.g. 'claude-opus-4-6' or 'openai/gpt-4o')",
    ),
    ("max_tokens", "Maximum output tokens per response"),
    ("verbose", "Enable verbose logging (true/false)"),
    (
        "permission_mode",
        "Permission mode: default | accept_edits | bypass_permissions | plan",
    ),
    (
        "agent_completion_policy",
        "Completion gate policy: enforce | warn | off",
    ),
    (
        "verification_policy",
        "Verification policy for changed code: auto | ask | off",
    ),
    (
        "agent_reliability_profile",
        "Reliability profile: standard | strict",
    ),
    ("agent_speed_profile", "Speed profile: balanced | fast_safe"),
    (
        "auto_compact",
        "Auto-compact conversation when context fills (true/false)",
    ),
];

fn normalize_model_id(model: &str) -> Option<String> {
    mangocode_api::normalize_model_id(model)
}

fn apply_model_setting(config: &mut Config, model: String) -> bool {
    mangocode_api::apply_model_selection_to_config(config, &model, None)
}

fn apply_model_setting_to_settings(
    settings: &mut Settings,
    runtime_provider: Option<&str>,
    model: String,
) -> bool {
    let Some(model) = normalize_model_id(&model) else {
        return false;
    };

    if settings.config.provider.is_none() {
        settings.config.provider = settings
            .provider
            .clone()
            .or_else(|| runtime_provider.map(ToOwned::to_owned));
    }

    apply_model_setting(&mut settings.config, model);

    if let Some(provider) = settings.config.provider.clone() {
        settings.provider = Some(provider);
    }
    true
}

fn effective_model_for_tool_config(config: &Config) -> String {
    let mut registry = mangocode_api::ModelRegistry::new();
    registry.load_standard_cache();
    mangocode_api::effective_model_for_config(config, &registry)
}

fn normalized_choice(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('-', "_")
}

fn parse_completion_policy(value: &str) -> Option<AgentCompletionPolicy> {
    match normalized_choice(value).as_str() {
        "enforce" | "enforced" | "on" | "enable" | "enabled" | "true" | "1" => {
            Some(AgentCompletionPolicy::Enforce)
        }
        "warn" | "warning" | "advisory" => Some(AgentCompletionPolicy::Warn),
        "off" | "disable" | "disabled" | "false" | "0" | "none" => Some(AgentCompletionPolicy::Off),
        _ => None,
    }
}

fn parse_verification_policy(value: &str) -> Option<VerificationPolicy> {
    match normalized_choice(value).as_str() {
        "auto" | "on" | "enable" | "enabled" | "true" | "1" => Some(VerificationPolicy::Auto),
        "ask" | "prompt" | "manual" => Some(VerificationPolicy::Ask),
        "off" | "disable" | "disabled" | "false" | "0" | "none" => Some(VerificationPolicy::Off),
        _ => None,
    }
}

fn parse_reliability_profile(value: &str) -> Option<AgentReliabilityProfile> {
    match normalized_choice(value).as_str() {
        "standard" | "balanced" | "normal" => Some(AgentReliabilityProfile::Standard),
        "strict" | "reliable" | "reliable_autonomy" | "default" => {
            Some(AgentReliabilityProfile::Strict)
        }
        _ => None,
    }
}

fn parse_speed_profile(value: &str) -> Option<AgentSpeedProfile> {
    match normalized_choice(value).as_str() {
        "balanced" | "balance" | "standard" => Some(AgentSpeedProfile::Balanced),
        "fast_safe" | "fast" | "safe_fast" | "default" => Some(AgentSpeedProfile::FastSafe),
        _ => None,
    }
}

#[async_trait]
impl Tool for ConfigTool {
    fn name(&self) -> &str {
        "Config"
    }

    fn description(&self) -> &str {
        "Get or set MangoCode configuration settings. Omit 'value' to read the current value. \
         Supported settings: model, max_tokens, verbose, permission_mode, agent_completion_policy, \
         verification_policy, agent_reliability_profile, agent_speed_profile, auto_compact. \
         Changes persist to ~/.mangocode/settings.json."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Write
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "setting": {
                    "type": "string",
                    "description": "Setting key (e.g. 'model', 'verbose', 'max_tokens', 'permission_mode', 'verification_policy')"
                },
                "value": {
                    "description": "New value to set. Omit to read the current value."
                }
            },
            "required": ["setting"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: ConfigInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        let key_raw = params.setting.trim();
        let key_normalized = key_raw.to_ascii_lowercase().replace('-', "_");
        let key = key_normalized.as_str();

        // List all supported settings
        if key == "list" || key == "help" {
            let lines: Vec<String> = SUPPORTED_SETTINGS
                .iter()
                .map(|(k, d)| format!("  {} — {}", k, d))
                .collect();
            return ToolResult::success(format!("Supported settings:\n{}", lines.join("\n")));
        }

        if params.value.is_some() {
            if let Err(e) = ctx.check_permission(self.name(), &format!("Set config {}", key), false)
            {
                return ToolResult::error(e.to_string());
            }
        }

        // Load current settings
        let mut settings = match Settings::load().await {
            Ok(s) => s,
            Err(e) => return ToolResult::error(format!("Failed to load settings: {}", e)),
        };

        if let Some(new_value) = params.value {
            // SET operation
            match key {
                "model" => {
                    let s = match new_value.as_str() {
                        Some(s) => match normalize_model_id(s) {
                            Some(model) => model,
                            None => {
                                return ToolResult::error(
                                    "'model' must be a non-empty string".to_string(),
                                );
                            }
                        },
                        None => return ToolResult::error("'model' must be a string".to_string()),
                    };
                    if !apply_model_setting_to_settings(
                        &mut settings,
                        ctx.config.provider.as_deref(),
                        s.clone(),
                    ) {
                        return ToolResult::error("'model' must be a non-empty string".to_string());
                    }
                    if let Err(e) = settings.save().await {
                        return ToolResult::error(format!("Failed to save settings: {}", e));
                    }
                    ToolResult::success(format!("model = \"{}\"", s))
                }
                "max_tokens" => {
                    let n = match new_value.as_u64() {
                        Some(n) => n as u32,
                        None => {
                            return ToolResult::error(
                                "'max_tokens' must be a positive integer".to_string(),
                            );
                        }
                    };
                    settings.config.max_tokens = Some(n);
                    if let Err(e) = settings.save().await {
                        return ToolResult::error(format!("Failed to save settings: {}", e));
                    }
                    ToolResult::success(format!("max_tokens = {}", n))
                }
                "verbose" => {
                    let b = match new_value.as_bool() {
                        Some(b) => b,
                        None => {
                            return ToolResult::error(
                                "'verbose' must be true or false".to_string(),
                            );
                        }
                    };
                    settings.config.verbose = b;
                    if let Err(e) = settings.save().await {
                        return ToolResult::error(format!("Failed to save settings: {}", e));
                    }
                    ToolResult::success(format!("verbose = {}", b))
                }
                "auto_compact" => {
                    let b = match new_value.as_bool() {
                        Some(b) => b,
                        None => {
                            return ToolResult::error(
                                "'auto_compact' must be true or false".to_string(),
                            );
                        }
                    };
                    settings.config.auto_compact = b;
                    if let Err(e) = settings.save().await {
                        return ToolResult::error(format!("Failed to save settings: {}", e));
                    }
                    ToolResult::success(format!("auto_compact = {}", b))
                }
                "permission_mode" => {
                    use mangocode_core::config::PermissionMode;
                    let s = match new_value.as_str() {
                        Some(s) => s,
                        None => {
                            return ToolResult::error(
                                "'permission_mode' must be a string".to_string(),
                            );
                        }
                    };
                    let mode = match s {
                        "default" => PermissionMode::Default,
                        "accept_edits" | "acceptEdits" => PermissionMode::AcceptEdits,
                        "bypass_permissions" | "bypassPermissions" => {
                            PermissionMode::BypassPermissions
                        }
                        "plan" => PermissionMode::Plan,
                        _ => {
                            return ToolResult::error(format!(
                                "Unknown permission_mode '{}'. Use: default | accept_edits | bypass_permissions | plan",
                                s
                            ));
                        }
                    };
                    settings.config.permission_mode = mode;
                    if let Err(e) = settings.save().await {
                        return ToolResult::error(format!("Failed to save settings: {}", e));
                    }
                    ToolResult::success(format!("permission_mode = \"{}\"", s))
                }
                "agent_completion_policy" | "completion_policy" => {
                    let s = match new_value.as_str() {
                        Some(s) => s,
                        None => {
                            return ToolResult::error(
                                "'agent_completion_policy' must be a string".to_string(),
                            );
                        }
                    };
                    let Some(policy) = parse_completion_policy(s) else {
                        return ToolResult::error(
                            "Unknown agent_completion_policy. Use: enforce | warn | off"
                                .to_string(),
                        );
                    };
                    settings.config.agent_completion_policy = policy;
                    if let Err(e) = settings.save().await {
                        return ToolResult::error(format!("Failed to save settings: {}", e));
                    }
                    ToolResult::success(format!("agent_completion_policy = \"{}\"", policy.label()))
                }
                "verification_policy" => {
                    let s = match new_value.as_str() {
                        Some(s) => s,
                        None => {
                            return ToolResult::error(
                                "'verification_policy' must be a string".to_string(),
                            );
                        }
                    };
                    let Some(policy) = parse_verification_policy(s) else {
                        return ToolResult::error(
                            "Unknown verification_policy. Use: auto | ask | off".to_string(),
                        );
                    };
                    settings.config.verification_policy = policy;
                    if let Err(e) = settings.save().await {
                        return ToolResult::error(format!("Failed to save settings: {}", e));
                    }
                    ToolResult::success(format!("verification_policy = \"{}\"", policy.label()))
                }
                "agent_reliability_profile" | "reliability_profile" => {
                    let s = match new_value.as_str() {
                        Some(s) => s,
                        None => {
                            return ToolResult::error(
                                "'agent_reliability_profile' must be a string".to_string(),
                            );
                        }
                    };
                    let Some(profile) = parse_reliability_profile(s) else {
                        return ToolResult::error(
                            "Unknown agent_reliability_profile. Use: standard | strict".to_string(),
                        );
                    };
                    settings.config.agent_reliability_profile = profile;
                    if let Err(e) = settings.save().await {
                        return ToolResult::error(format!("Failed to save settings: {}", e));
                    }
                    ToolResult::success(format!(
                        "agent_reliability_profile = \"{}\"",
                        profile.label()
                    ))
                }
                "agent_speed_profile" | "speed_profile" => {
                    let s = match new_value.as_str() {
                        Some(s) => s,
                        None => {
                            return ToolResult::error(
                                "'agent_speed_profile' must be a string".to_string(),
                            );
                        }
                    };
                    let Some(profile) = parse_speed_profile(s) else {
                        return ToolResult::error(
                            "Unknown agent_speed_profile. Use: balanced | fast_safe".to_string(),
                        );
                    };
                    settings.config.agent_speed_profile = profile;
                    if let Err(e) = settings.save().await {
                        return ToolResult::error(format!("Failed to save settings: {}", e));
                    }
                    ToolResult::success(format!("agent_speed_profile = \"{}\"", profile.label()))
                }
                _ => ToolResult::error(format!(
                    "Unknown setting '{}'. Use setting='list' to see all supported settings.",
                    key
                )),
            }
        } else {
            // GET operation — report the live session config (`ctx.config`),
            // which carries session-only overrides (CLI `--model`, in-session
            // `/model`) that are never written to settings.json. Reading the
            // on-disk settings here would report the stale persisted default and
            // mislead the agent about the model actually in effect.
            runtime_setting_value(key, &ctx.config).unwrap_or_else(|| {
                ToolResult::error(format!(
                    "Unknown setting '{}'. Use setting='list' to see all supported settings.",
                    key
                ))
            })
        }
    }
}

fn runtime_setting_value(key: &str, config: &Config) -> Option<ToolResult> {
    match key {
        "model" => Some(ToolResult::success(format!(
            "model = \"{}\"",
            effective_model_for_tool_config(config)
        ))),
        "max_tokens" => Some(ToolResult::success(format!(
            "max_tokens = {}",
            config.effective_max_tokens()
        ))),
        "verbose" => Some(ToolResult::success(format!("verbose = {}", config.verbose))),
        "auto_compact" => Some(ToolResult::success(format!(
            "auto_compact = {}",
            config.auto_compact
        ))),
        "permission_mode" => Some(ToolResult::success(format!(
            "permission_mode = \"{}\"",
            permission_mode_str(&config.permission_mode)
        ))),
        "agent_completion_policy" | "completion_policy" => Some(ToolResult::success(format!(
            "agent_completion_policy = \"{}\"",
            config.agent_completion_policy.label()
        ))),
        "verification_policy" => Some(ToolResult::success(format!(
            "verification_policy = \"{}\"",
            config.verification_policy.label()
        ))),
        "agent_reliability_profile" | "reliability_profile" => Some(ToolResult::success(format!(
            "agent_reliability_profile = \"{}\"",
            config.agent_reliability_profile.label()
        ))),
        "agent_speed_profile" | "speed_profile" => Some(ToolResult::success(format!(
            "agent_speed_profile = \"{}\"",
            config.agent_speed_profile.label()
        ))),
        _ => None,
    }
}

fn permission_mode_str(mode: &mangocode_core::config::PermissionMode) -> &'static str {
    use mangocode_core::config::PermissionMode;
    match mode {
        PermissionMode::Default => "default",
        PermissionMode::AcceptEdits => "accept_edits",
        PermissionMode::BypassPermissions => "bypass_permissions",
        PermissionMode::Plan => "plan",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_setting_updates_provider_for_prefixed_gateway_model() {
        let mut config = Config::default();

        assert!(apply_model_setting(
            &mut config,
            " openrouter/anthropic/claude-sonnet-4 ".to_string(),
        ));

        assert_eq!(config.provider.as_deref(), Some("openrouter"));
        assert_eq!(
            config.model.as_deref(),
            Some("openrouter/anthropic/claude-sonnet-4")
        );
    }

    #[test]
    fn model_setting_ignores_blank_model() {
        let mut config = Config {
            provider: Some("anthropic".to_string()),
            model: Some("claude-haiku-4-5".to_string()),
            ..Default::default()
        };

        assert!(!apply_model_setting(&mut config, "   ".to_string()));

        assert_eq!(config.provider.as_deref(), Some("anthropic"));
        assert_eq!(config.model.as_deref(), Some("claude-haiku-4-5"));
    }

    #[test]
    fn model_setting_strips_anthropic_provider_prefix() {
        let mut config = Config::default();

        apply_model_setting(&mut config, "anthropic/claude-sonnet-4".to_string());

        assert_eq!(config.provider.as_deref(), Some("anthropic"));
        assert_eq!(config.model.as_deref(), Some("claude-sonnet-4"));
    }

    #[test]
    fn model_setting_infers_provider_for_unprefixed_known_model() {
        let mut config = Config::default();

        apply_model_setting(&mut config, "gpt-4o".to_string());

        assert_eq!(config.provider.as_deref(), Some("openai"));
        assert_eq!(config.model.as_deref(), Some("gpt-4o"));
    }

    #[test]
    fn model_setting_preserves_existing_provider_for_unknown_namespaced_model() {
        let mut config = Config {
            provider: Some("openrouter".to_string()),
            ..Default::default()
        };

        apply_model_setting(&mut config, "meta-llama/Llama-3.3-70B".to_string());

        assert_eq!(config.provider.as_deref(), Some("openrouter"));
        assert_eq!(config.model.as_deref(), Some("meta-llama/Llama-3.3-70B"));
    }

    #[test]
    fn model_setting_to_settings_preserves_top_level_provider() {
        let mut settings = Settings {
            provider: Some("openrouter".to_string()),
            ..Default::default()
        };

        assert!(apply_model_setting_to_settings(
            &mut settings,
            None,
            "meta-llama/Llama-3.3-70B".to_string(),
        ));

        assert_eq!(settings.provider.as_deref(), Some("openrouter"));
        assert_eq!(settings.config.provider.as_deref(), Some("openrouter"));
        assert_eq!(
            settings.config.model.as_deref(),
            Some("meta-llama/Llama-3.3-70B")
        );
    }

    #[test]
    fn model_setting_to_settings_ignores_blank_model() {
        let mut settings = Settings {
            provider: Some("openrouter".to_string()),
            config: Config {
                provider: Some("openrouter".to_string()),
                model: Some("openai/gpt-4o".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };

        assert!(!apply_model_setting_to_settings(
            &mut settings,
            Some("openai"),
            "   ".to_string(),
        ));

        assert_eq!(settings.provider.as_deref(), Some("openrouter"));
        assert_eq!(settings.config.provider.as_deref(), Some("openrouter"));
        assert_eq!(settings.config.model.as_deref(), Some("openai/gpt-4o"));
    }

    #[test]
    fn reliability_setting_parsers_accept_runtime_aliases() {
        assert_eq!(
            parse_completion_policy("enforce"),
            Some(AgentCompletionPolicy::Enforce)
        );
        assert_eq!(
            parse_completion_policy("warn"),
            Some(AgentCompletionPolicy::Warn)
        );
        assert_eq!(
            parse_completion_policy("off"),
            Some(AgentCompletionPolicy::Off)
        );
        assert_eq!(
            parse_verification_policy("auto"),
            Some(VerificationPolicy::Auto)
        );
        assert_eq!(
            parse_verification_policy("ask"),
            Some(VerificationPolicy::Ask)
        );
        assert_eq!(
            parse_verification_policy("off"),
            Some(VerificationPolicy::Off)
        );
        assert_eq!(
            parse_reliability_profile("reliable-autonomy"),
            Some(AgentReliabilityProfile::Strict)
        );
        assert_eq!(
            parse_reliability_profile("standard"),
            Some(AgentReliabilityProfile::Standard)
        );
        assert_eq!(
            parse_speed_profile("fast-safe"),
            Some(AgentSpeedProfile::FastSafe)
        );
        assert_eq!(
            parse_speed_profile("default"),
            Some(AgentSpeedProfile::FastSafe)
        );
        assert_eq!(
            parse_speed_profile("balanced"),
            Some(AgentSpeedProfile::Balanced)
        );
    }

    #[test]
    fn runtime_get_uses_active_reliability_policy_config() {
        let config = Config {
            agent_completion_policy: AgentCompletionPolicy::Warn,
            verification_policy: VerificationPolicy::Ask,
            agent_reliability_profile: AgentReliabilityProfile::Standard,
            agent_speed_profile: AgentSpeedProfile::Balanced,
            ..Default::default()
        };

        assert_eq!(
            runtime_setting_value("agent_completion_policy", &config).map(|result| result.content),
            Some("agent_completion_policy = \"warn\"".to_string())
        );
        assert_eq!(
            runtime_setting_value("verification_policy", &config).map(|result| result.content),
            Some("verification_policy = \"ask\"".to_string())
        );
        assert_eq!(
            runtime_setting_value("agent_reliability_profile", &config)
                .map(|result| result.content),
            Some("agent_reliability_profile = \"standard\"".to_string())
        );
        assert_eq!(
            runtime_setting_value("agent_speed_profile", &config).map(|result| result.content),
            Some("agent_speed_profile = \"balanced\"".to_string())
        );
    }
}
