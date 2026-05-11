// ConfigTool: get or set MangoCode configuration settings at runtime.
//
// Reads from and persists to ~/.mangocode/settings.json.
// Supported settings: model, max_tokens, verbose, permission_mode.

use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use mangocode_core::config::{Config, Settings};
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

#[async_trait]
impl Tool for ConfigTool {
    fn name(&self) -> &str {
        "Config"
    }

    fn description(&self) -> &str {
        "Get or set MangoCode configuration settings. Omit 'value' to read the current value. \
         Supported settings: model, max_tokens, verbose, permission_mode, auto_compact. \
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
                    "description": "Setting key (e.g. 'model', 'verbose', 'max_tokens', 'permission_mode')"
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

        let key = params.setting.trim();

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
                _ => ToolResult::error(format!(
                    "Unknown setting '{}'. Use setting='list' to see all supported settings.",
                    key
                )),
            }
        } else {
            // GET operation
            match key {
                "model" => ToolResult::success(format!(
                    "model = \"{}\"",
                    effective_model_for_tool_config(&ctx.config)
                )),
                "max_tokens" => ToolResult::success(format!(
                    "max_tokens = {}",
                    settings.config.effective_max_tokens()
                )),
                "verbose" => ToolResult::success(format!("verbose = {}", settings.config.verbose)),
                "auto_compact" => {
                    ToolResult::success(format!("auto_compact = {}", settings.config.auto_compact))
                }
                "permission_mode" => ToolResult::success(format!(
                    "permission_mode = \"{}\"",
                    permission_mode_str(&settings.config.permission_mode)
                )),
                _ => ToolResult::error(format!(
                    "Unknown setting '{}'. Use setting='list' to see all supported settings.",
                    key
                )),
            }
        }
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
}
