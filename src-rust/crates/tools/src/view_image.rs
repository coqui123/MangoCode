use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Debug, Clone, Deserialize)]
struct ViewImageInput {
    path: String,
    #[serde(default)]
    label: Option<String>,
}

pub struct ViewImageTool;

#[async_trait]
impl Tool for ViewImageTool {
    fn name(&self) -> &str {
        "ViewImage"
    }

    fn description(&self) -> &str {
        "Inspect a local image file by path and return image metadata plus OCR text when available. Alias: view_image."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the image file, absolute or relative to the workspace"
                },
                "label": {
                    "type": "string",
                    "description": "Optional display label for the image"
                }
            },
            "required": ["path"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: ViewImageInput = match serde_json::from_value(input) {
            Ok(params) => params,
            Err(err) => return ToolResult::error(format!("Invalid ViewImage input: {err}")),
        };

        let path = ctx.resolve_path(&params.path);
        let description = format!("view image {}", path.display());
        if let Err(err) = ctx.check_permission(self.name(), &description, true) {
            return ToolResult::error(err.to_string());
        }
        if !path.exists() {
            return ToolResult::error(format!("Image not found: {}", path.display()));
        }

        match mangocode_core::smart_attachments::image_markdown_with_ocr_fallback(
            &path,
            params.label.as_deref(),
            Some(&ctx.config.attachments),
        ) {
            Ok(markdown) => ToolResult::success(markdown),
            Err(err) => ToolResult::error(format!("Failed to inspect image: {err}")),
        }
    }
}
