// NotebookEditTool: edit Jupyter notebook cells (.ipynb files).
//
// Supports three edit modes:
//   - replace: modify an existing cell's source
//   - insert: add a new cell after a given cell (or at the start)
//   - delete: remove a cell

use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::debug;

pub struct NotebookEditTool;

#[derive(Debug, Deserialize)]
struct NotebookEditInput {
    notebook_path: String,
    #[serde(default)]
    cell_id: Option<String>,
    #[serde(default)]
    new_source: Option<String>,
    #[serde(default = "default_cell_type")]
    cell_type: String,
    #[serde(default = "default_edit_mode")]
    edit_mode: String,
    #[serde(default)]
    confirm_conflicts: bool,
}

fn default_cell_type() -> String {
    "code".to_string()
}

fn default_edit_mode() -> String {
    "replace".to_string()
}

fn normalize_notebook_cell_type(value: &str) -> Result<&'static str, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "code" => Ok("code"),
        "markdown" => Ok("markdown"),
        _ => {
            let value =
                serde_json::to_string(value).unwrap_or_else(|_| "\"<invalid>\"".to_string());
            Err(format!(
                "Invalid cell_type: {value}. Expected one of: code, markdown"
            ))
        }
    }
}

fn normalize_notebook_edit_mode(value: &str) -> Result<&'static str, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "replace" => Ok("replace"),
        "insert" => Ok("insert"),
        "delete" => Ok("delete"),
        _ => {
            let value =
                serde_json::to_string(value).unwrap_or_else(|_| "\"<invalid>\"".to_string());
            Err(format!(
                "Invalid edit_mode: {value}. Expected one of: replace, insert, delete"
            ))
        }
    }
}

#[async_trait]
impl Tool for NotebookEditTool {
    fn name(&self) -> &str {
        mangocode_core::constants::TOOL_NAME_NOTEBOOK_EDIT
    }

    fn description(&self) -> &str {
        "Edit cells in a Jupyter notebook (.ipynb file). Supports three edit modes:\n\
         - replace: modify an existing cell's source (requires cell_id)\n\
         - insert: add a new cell after a given cell (or at the start if no cell_id)\n\
         - delete: remove a cell (requires cell_id)\n\
         You MUST read the notebook file before editing."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Write
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "notebook_path": {
                    "type": "string",
                    "description": "Absolute path to the .ipynb notebook file"
                },
                "cell_id": {
                    "type": "string",
                    "description": "Cell ID (UUID or 'cell-N' index). Required for replace/delete."
                },
                "new_source": {
                    "type": "string",
                    "description": "New cell content. Required for replace/insert."
                },
                "cell_type": {
                    "type": "string",
                    "enum": ["code", "markdown"],
                    "description": "Cell type for insert operations (default: code)"
                },
                "edit_mode": {
                    "type": "string",
                    "enum": ["replace", "insert", "delete"],
                    "description": "Edit mode: replace, insert, or delete (default: replace)"
                },
                "confirm_conflicts": {
                    "type": "boolean",
                    "description": "Set true only after acknowledging active MangoCode coordination conflicts for this notebook."
                }
            },
            "required": ["notebook_path"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: NotebookEditInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };
        let edit_mode = match normalize_notebook_edit_mode(&params.edit_mode) {
            Ok(mode) => mode,
            Err(e) => return ToolResult::error(e),
        };
        let cell_type = match normalize_notebook_cell_type(&params.cell_type) {
            Ok(cell_type) => cell_type,
            Err(e) => return ToolResult::error(e),
        };

        let path = ctx.resolve_path(&params.notebook_path);

        // Validate extension
        if path.extension().and_then(|e| e.to_str()) != Some("ipynb") {
            return ToolResult::error("File must have .ipynb extension".to_string());
        }

        // Permission check
        if let Err(e) = ctx.check_permission(
            self.name(),
            &format!("Edit notebook {}", path.display()),
            false,
        ) {
            return ToolResult::error(e.to_string());
        }

        let coordination_conflicts = match crate::coordination::preflight_write_conflicts(
            ctx,
            self.name(),
            std::slice::from_ref(&path),
            params.confirm_conflicts,
        ) {
            Ok(conflicts) => conflicts,
            Err(result) => return result,
        };
        let _coordination_write_claim = match crate::coordination::begin_transient_write_claim(
            ctx,
            self.name(),
            std::slice::from_ref(&path),
            params.confirm_conflicts,
        ) {
            Ok(guard) => guard,
            Err(result) => return result,
        };

        // Read notebook
        let content = match tokio::fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => return ToolResult::error(format!("Failed to read notebook: {}", e)),
        };

        let mut notebook: Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(e) => return ToolResult::error(format!("Invalid notebook JSON: {}", e)),
        };

        debug!(path = %path.display(), mode = %edit_mode, "Editing notebook");

        let result = match edit_mode {
            "replace" => {
                let cell_id = match &params.cell_id {
                    Some(id) => id.clone(),
                    None => {
                        return ToolResult::error(
                            "cell_id is required for replace mode".to_string(),
                        );
                    }
                };
                let new_source = match &params.new_source {
                    Some(s) => s.clone(),
                    None => {
                        return ToolResult::error(
                            "new_source is required for replace mode".to_string(),
                        );
                    }
                };
                replace_cell(&mut notebook, &cell_id, &new_source)
            }
            "insert" => {
                let new_source = match &params.new_source {
                    Some(s) => s.clone(),
                    None => {
                        return ToolResult::error(
                            "new_source is required for insert mode".to_string(),
                        );
                    }
                };
                insert_cell(
                    &mut notebook,
                    params.cell_id.as_deref(),
                    &new_source,
                    cell_type,
                )
            }
            "delete" => {
                let cell_id = match &params.cell_id {
                    Some(id) => id.clone(),
                    None => {
                        return ToolResult::error(
                            "cell_id is required for delete mode".to_string(),
                        );
                    }
                };
                delete_cell(&mut notebook, &cell_id)
            }
            _ => unreachable!("edit_mode is validated before editing"),
        };

        match result {
            Ok(msg) => {
                // Write back
                let updated = match serde_json::to_string_pretty(&notebook) {
                    Ok(s) => s,
                    Err(e) => {
                        return ToolResult::error(format!("Failed to serialize notebook: {}", e));
                    }
                };
                if let Err(e) = tokio::fs::write(&path, &updated).await {
                    return ToolResult::error(format!("Failed to write notebook: {}", e));
                }
                ctx.record_file_change(
                    path.clone(),
                    content.as_bytes(),
                    updated.as_bytes(),
                    self.name(),
                );
                let msg = crate::coordination::append_confirmed_conflict_note(
                    msg,
                    coordination_conflicts.as_deref(),
                );
                ToolResult::success(msg).with_metadata(json!({
                    "notebook_path": path.display().to_string(),
                    "edit_mode": edit_mode,
                    "cell_type": cell_type,
                    "coordination_conflicts": coordination_conflicts,
                }))
            }
            Err(e) => ToolResult::error(e),
        }
    }
}

// ---------------------------------------------------------------------------
// Notebook manipulation helpers
// ---------------------------------------------------------------------------

/// Resolve a cell index from "cell-N" notation or return `None` for UUID lookup.
fn parse_cell_index(cell_id: &str) -> Option<usize> {
    cell_id
        .strip_prefix("cell-")
        .and_then(|n| n.parse::<usize>().ok())
}

/// Find the position of a cell in the `cells` array by id or "cell-N".
fn find_cell_index(cells: &[Value], cell_id: &str) -> Result<usize, String> {
    // Try "cell-N" index format first
    if let Some(idx) = parse_cell_index(cell_id) {
        if idx < cells.len() {
            return Ok(idx);
        }
        return Err(format!(
            "Cell index {} is out of range (notebook has {} cells)",
            idx,
            cells.len()
        ));
    }

    // Try UUID match
    for (i, cell) in cells.iter().enumerate() {
        if let Some(id) = cell.get("id").and_then(|v| v.as_str()) {
            if id == cell_id {
                return Ok(i);
            }
        }
    }

    Err(format!("Cell '{}' not found", cell_id))
}

/// Generate a simple random cell ID (8 hex chars, like nbformat ≥ 4.5).
fn generate_cell_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    format!("{:08x}", nanos ^ 0xdeadbeef_u32)
}

/// Build a new cell JSON object.
fn make_cell(cell_type: &str, source: &str, cell_id: &str) -> Value {
    let source_lines: Vec<Value> = if source.is_empty() {
        vec![]
    } else {
        let lines: Vec<&str> = source.split_inclusive('\n').collect();
        lines.iter().map(|l| Value::String(l.to_string())).collect()
    };

    match cell_type {
        "markdown" => json!({
            "cell_type": "markdown",
            "id": cell_id,
            "metadata": {},
            "source": source_lines
        }),
        _ => json!({
            "cell_type": "code",
            "id": cell_id,
            "metadata": {},
            "source": source_lines,
            "outputs": [],
            "execution_count": null
        }),
    }
}

fn replace_cell(notebook: &mut Value, cell_id: &str, new_source: &str) -> Result<String, String> {
    let cells = notebook
        .get_mut("cells")
        .and_then(|c| c.as_array_mut())
        .ok_or_else(|| "Notebook has no 'cells' array".to_string())?;

    let idx = find_cell_index(cells, cell_id)?;

    let cell = &mut cells[idx];
    let source_lines: Vec<Value> = new_source
        .split_inclusive('\n')
        .map(|l| Value::String(l.to_string()))
        .collect();

    cell["source"] = Value::Array(source_lines);

    // Reset execution state for code cells
    if cell.get("cell_type").and_then(|t| t.as_str()) == Some("code") {
        cell["outputs"] = Value::Array(vec![]);
        cell["execution_count"] = Value::Null;
    }

    Ok(format!("Replaced cell '{}' (index {})", cell_id, idx))
}

fn insert_cell(
    notebook: &mut Value,
    after_cell_id: Option<&str>,
    new_source: &str,
    cell_type: &str,
) -> Result<String, String> {
    let cells = notebook
        .get_mut("cells")
        .and_then(|c| c.as_array_mut())
        .ok_or_else(|| "Notebook has no 'cells' array".to_string())?;

    let insert_at = if let Some(id) = after_cell_id {
        find_cell_index(cells, id)? + 1
    } else {
        0
    };

    let new_id = generate_cell_id();
    let cell = make_cell(cell_type, new_source, &new_id);

    cells.insert(insert_at, cell);
    Ok(format!(
        "Inserted {} cell '{}' at position {}",
        cell_type, new_id, insert_at
    ))
}

fn delete_cell(notebook: &mut Value, cell_id: &str) -> Result<String, String> {
    let cells = notebook
        .get_mut("cells")
        .and_then(|c| c.as_array_mut())
        .ok_or_else(|| "Notebook has no 'cells' array".to_string())?;

    let idx = find_cell_index(cells, cell_id)?;
    cells.remove(idx);

    Ok(format!("Deleted cell '{}' (was at index {})", cell_id, idx))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notebook_modes_are_normalized_or_rejected() {
        assert_eq!(normalize_notebook_edit_mode(" INSERT "), Ok("insert"));
        assert_eq!(normalize_notebook_cell_type(" Markdown "), Ok("markdown"));

        let mode_err = normalize_notebook_edit_mode("replace\ndelete")
            .expect_err("invalid edit mode must be rejected");
        let cell_err = normalize_notebook_cell_type("code\nmarkdown")
            .expect_err("invalid cell type must be rejected");

        assert!(mode_err.contains("Invalid edit_mode"));
        assert!(mode_err.contains("\"replace\\ndelete\""));
        assert!(cell_err.contains("Invalid cell_type"));
        assert!(cell_err.contains("\"code\\nmarkdown\""));
    }
}
