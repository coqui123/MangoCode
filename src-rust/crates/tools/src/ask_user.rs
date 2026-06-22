// AskUserQuestion tool: ask the human operator one or more clarifying
// questions and wait for their response. Each question may have a header, optional
// multiple-choice options (label + description), and a multi-select flag.
// A free-form "Other" answer is always implicitly available.

#[cfg(feature = "tool-ask-user")]
use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
#[cfg(feature = "tool-ask-user")]
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
#[cfg(feature = "tool-ask-user")]
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};
#[cfg(feature = "tool-ask-user")]
use tracing::debug;

// ---------------------------------------------------------------------------
// Public types: prompt + response payloads sent over the question channel.
// ---------------------------------------------------------------------------

/// A single option offered for one question. Shown to the user as a
/// selectable list entry; `description` is rendered in dimmer text below
/// the label if present.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionOption {
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// One clarification question. The TUI walks through questions in order;
/// each one collects either an option set or a free-form "Other" response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Question {
    /// Full question text shown to the user.
    pub question: String,
    /// Optional short noun-phrase shown as a header chip above the question.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub header: Option<String>,
    /// If true, the user may select multiple options for this question.
    #[serde(default)]
    pub multi_select: bool,
    /// Selectable options. May be empty (user must answer via "Other").
    #[serde(default)]
    pub options: Vec<QuestionOption>,
}

/// A prompt sent from the AskUserQuestion tool to the TUI. The TUI walks
/// the user through each question and posts back a [`QuestionResponse`].
pub struct QuestionPrompt {
    pub tool_use_id: String,
    pub questions: Vec<Question>,
    pub response_tx: oneshot::Sender<QuestionResponse>,
}

/// The user's answer to a single question.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum QuestionAnswer {
    /// User selected one or more provided options (by label).
    Options { selected: Vec<String> },
    /// User typed a free-form response instead of picking an option.
    Other { text: String },
    /// User cancelled / dismissed the dialog without answering.
    Cancelled,
}

impl QuestionAnswer {
    /// Render this answer as plain text for the model.
    pub fn render_plain(&self) -> String {
        match self {
            QuestionAnswer::Options { selected } => {
                if selected.is_empty() {
                    "(no option selected)".to_string()
                } else {
                    selected.join(", ")
                }
            }
            QuestionAnswer::Other { text } => text.clone(),
            QuestionAnswer::Cancelled => "(user cancelled)".to_string(),
        }
    }
}

/// The aggregated response posted back when every question has been
/// answered (or the dialog was cancelled).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionResponse {
    /// One answer per question, indexed by the question's position.
    pub answers: Vec<QuestionAnswer>,
    /// True if the user pressed Esc / dismissed before completing every
    /// question. The model gets a clear "cancelled" signal so it can
    /// re-ask, fall back to assumptions, or wait for new input.
    pub cancelled: bool,
}

/// Channel handle stored on `ToolContext`. Tools clone the sender to post
/// a [`QuestionPrompt`] and await the response over the oneshot inside it.
pub type QuestionPromptSender = mpsc::UnboundedSender<QuestionPrompt>;

// ---------------------------------------------------------------------------
// Input-schema types: what the model is asked to provide.
// ---------------------------------------------------------------------------

#[cfg(feature = "tool-ask-user")]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AskUserInput {
    questions: Vec<InputQuestion>,
}

#[cfg(feature = "tool-ask-user")]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputQuestion {
    question: String,
    #[serde(default)]
    header: Option<String>,
    /// Claude uses `multiSelect`; we accept both `multiSelect` and
    /// `multi_select` so model output in either convention parses.
    #[serde(default, alias = "multiSelect")]
    multi_select: bool,
    #[serde(default)]
    options: Vec<InputOption>,
}

#[cfg(feature = "tool-ask-user")]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputOption {
    label: String,
    #[serde(default)]
    description: Option<String>,
}

// ---------------------------------------------------------------------------
// Tool implementation.
// ---------------------------------------------------------------------------

#[cfg(feature = "tool-ask-user")]
pub struct AskUserQuestionTool;

#[cfg(feature = "tool-ask-user")]
#[async_trait]
impl Tool for AskUserQuestionTool {
    fn name(&self) -> &str {
        mangocode_core::constants::TOOL_NAME_ASK_USER
    }

    fn description(&self) -> &str {
        "Ask the user a clarifying question (or a small batch of related questions) \
         and wait for their answer. Use this when the request is genuinely \
         ambiguous and you cannot pick a sensible default. Each question may \
         include multiple-choice options (label + optional description) and a \
         multiSelect flag. Users can always type a free-form answer instead of \
         picking an option."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::None
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "description": "1-4 clarifying questions to ask. Keep each tightly scoped.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The full question text shown to the user."
                            },
                            "header": {
                                "type": "string",
                                "description": "Optional short label (under 12 chars) shown as a chip above the question, e.g. 'Auth method' or 'Library'."
                            },
                            "multiSelect": {
                                "type": "boolean",
                                "default": false,
                                "description": "If true, the user may pick multiple options for this question."
                            },
                            "options": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 4,
                                "description": "2-4 mutually-exclusive options (unless multiSelect). Users can always type 'Other' instead.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {
                                            "type": "string",
                                            "description": "Short option label (1-5 words)."
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": "Optional context about what this option means or its trade-offs."
                                        }
                                    },
                                    "required": ["label"]
                                }
                            }
                        },
                        "required": ["question", "options"]
                    }
                }
            },
            "required": ["questions"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let parsed: AskUserInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid AskUserQuestion input: {}", e)),
        };

        if parsed.questions.is_empty() {
            return ToolResult::error("AskUserQuestion needs at least one question".to_string());
        }
        if parsed.questions.len() > 4 {
            return ToolResult::error(
                "AskUserQuestion accepts at most 4 questions per call".to_string(),
            );
        }

        // Convert the parsed input into the question structures shipped
        // to the TUI.
        let questions: Vec<Question> = parsed
            .questions
            .into_iter()
            .map(|q| Question {
                question: q.question,
                header: q
                    .header
                    .map(|h| h.trim().to_string())
                    .filter(|h| !h.is_empty()),
                multi_select: q.multi_select,
                options: q
                    .options
                    .into_iter()
                    .map(|o| QuestionOption {
                        label: o.label,
                        description: o
                            .description
                            .map(|d| d.trim().to_string())
                            .filter(|d| !d.is_empty()),
                    })
                    .collect(),
            })
            .collect();

        debug!(
            num_questions = questions.len(),
            "AskUserQuestion: dispatching prompt"
        );

        if ctx.non_interactive {
            return ToolResult::error(
                "Cannot ask the user clarifying questions in non-interactive mode. \
                 Make a best-effort assumption and proceed."
                    .to_string(),
            );
        }

        let Some(tx) = ctx.question_prompt_tx.as_ref() else {
            return ToolResult::error(
                "AskUserQuestion is not available in this context (no HITL channel \
                 wired — e.g. running inside a sub-agent, proactive job, or ACP \
                 session). Make a best-effort assumption and proceed."
                    .to_string(),
            );
        };

        // A tool_use_id isn't directly available here, but it's not load-bearing
        // for the dialog flow — the TUI just needs a stable handle for tracing.
        let tool_use_id = format!("ask_user_{}", uuid::Uuid::new_v4());

        let (resp_tx, resp_rx) = oneshot::channel();
        let prompt = QuestionPrompt {
            tool_use_id: tool_use_id.clone(),
            questions: questions.clone(),
            response_tx: resp_tx,
        };
        if tx.send(prompt).is_err() {
            // Receiver dropped: TUI exited or crashed mid-question. Surface
            // a clean error so the model doesn't hang.
            return ToolResult::error(
                "AskUserQuestion channel closed before the user could respond.".to_string(),
            );
        }

        let response = match resp_rx.await {
            Ok(r) => r,
            Err(_) => {
                return ToolResult::error(
                    "AskUserQuestion response channel dropped before an answer was received."
                        .to_string(),
                );
            }
        };

        // Build a human-readable transcript for the model: one section per
        // question with the answer plainly stated.
        let mut transcript = String::new();
        for (i, q) in questions.iter().enumerate() {
            let ans = response
                .answers
                .get(i)
                .map(|a| a.render_plain())
                .unwrap_or_else(|| "(no answer)".to_string());
            if let Some(header) = &q.header {
                transcript.push_str(&format!("[{}] ", header));
            }
            transcript.push_str(&format!("Q: {}\nA: {}\n\n", q.question, ans));
        }
        if response.cancelled {
            transcript.push_str("(User dismissed the dialog before answering every question.)\n");
        }
        let transcript = transcript.trim_end().to_string();

        // Structured metadata so the TUI / harness can render a tidy log row.
        let meta = json!({
            "type": "ask_user",
            "tool_use_id": tool_use_id,
            "questions": questions.iter().map(|q| {
                json!({
                    "question": q.question,
                    "header": q.header,
                    "multi_select": q.multi_select,
                    "options": q.options.iter().map(|o| {
                        json!({ "label": o.label, "description": o.description })
                    }).collect::<Vec<_>>(),
                })
            }).collect::<Vec<_>>(),
            "answers": response.answers,
            "cancelled": response.cancelled,
        });

        ToolResult::success(transcript).with_metadata(meta)
    }
}
