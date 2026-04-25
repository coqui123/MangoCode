use async_trait::async_trait;
use futures::{stream, Stream};
use mangocode_core::provider_id::{ModelId, ProviderId};
use mangocode_core::types::{ContentBlock, UsageInfo};
use serde_json::Value;
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::provider::{LlmProvider, ModelInfo};
use crate::provider_error::ProviderError;
use crate::provider_types::{
    ProviderCapabilities, ProviderRequest, ProviderResponse, ProviderStatus, StopReason,
    StreamEvent, SystemPromptStyle,
};

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: Value,
}

impl ToolCall {
    pub fn new(id: impl Into<String>, name: impl Into<String>, input: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            input,
        }
    }
}

#[derive(Clone)]
pub struct MockProvider {
    responses: Arc<Mutex<VecDeque<String>>>,
    tool_calls: Arc<Mutex<VecDeque<Vec<ToolCall>>>>,
    request_log: Arc<Mutex<Vec<ProviderRequest>>>,
    next_id: Arc<AtomicUsize>,
    id: ProviderId,
}

impl MockProvider {
    pub fn with_responses(responses: Vec<&str>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(
                responses.into_iter().map(str::to_string).collect(),
            )),
            tool_calls: Arc::new(Mutex::new(VecDeque::new())),
            request_log: Arc::new(Mutex::new(Vec::new())),
            next_id: Arc::new(AtomicUsize::new(1)),
            id: ProviderId::new("mock"),
        }
    }

    pub fn with_tool_sequence(mut self, calls: Vec<Vec<ToolCall>>) -> Self {
        self.tool_calls = Arc::new(Mutex::new(calls.into_iter().collect()));
        self
    }

    pub fn request_count(&self) -> usize {
        self.request_log.lock().expect("request_log lock").len()
    }

    fn next_turn(&self, request: ProviderRequest) -> (String, Vec<ToolCall>, String, UsageInfo) {
        self.request_log
            .lock()
            .expect("request_log lock")
            .push(request);

        let id_num = self.next_id.fetch_add(1, Ordering::Relaxed);
        let id = format!("mock-msg-{}", id_num);

        let tools = self
            .tool_calls
            .lock()
            .expect("tool_calls lock")
            .pop_front()
            .unwrap_or_default();

        let text = if tools.is_empty() {
            self.responses
                .lock()
                .expect("responses lock")
                .pop_front()
                .unwrap_or_else(|| "mock response".to_string())
        } else {
            String::new()
        };

        (
            id,
            tools,
            text,
            UsageInfo {
                input_tokens: 10,
                output_tokens: 5,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        )
    }
}

#[async_trait]
impl LlmProvider for MockProvider {
    fn id(&self) -> &ProviderId {
        &self.id
    }

    fn name(&self) -> &str {
        "Mock Provider"
    }

    async fn create_message(
        &self,
        request: ProviderRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        let (id, tools, text, usage) = self.next_turn(request);
        let (content, stop_reason) = if tools.is_empty() {
            (vec![ContentBlock::Text { text }], StopReason::EndTurn)
        } else {
            (
                tools
                    .into_iter()
                    .map(|c| ContentBlock::ToolUse {
                        id: c.id,
                        name: c.name,
                        input: c.input,
                    })
                    .collect(),
                StopReason::ToolUse,
            )
        };

        Ok(ProviderResponse {
            id,
            content,
            stop_reason,
            usage,
            model: "mock-model".to_string(),
        })
    }

    async fn create_message_stream(
        &self,
        request: ProviderRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        let (id, tools, text, usage) = self.next_turn(request);

        let mut events: Vec<Result<StreamEvent, ProviderError>> = Vec::new();
        events.push(Ok(StreamEvent::MessageStart {
            id,
            model: "mock-model".to_string(),
            usage: usage.clone(),
        }));

        if tools.is_empty() {
            events.push(Ok(StreamEvent::TextDelta { index: 0, text }));
            events.push(Ok(StreamEvent::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: Some(usage),
            }));
        } else {
            for (index, tool) in tools.into_iter().enumerate() {
                events.push(Ok(StreamEvent::ContentBlockStart {
                    index,
                    content_block: ContentBlock::ToolUse {
                        id: tool.id,
                        name: tool.name,
                        input: serde_json::json!({}),
                    },
                }));
                events.push(Ok(StreamEvent::InputJsonDelta {
                    index,
                    partial_json: tool.input.to_string(),
                }));
                events.push(Ok(StreamEvent::ContentBlockStop { index }));
            }

            events.push(Ok(StreamEvent::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: Some(usage),
            }));
        }

        events.push(Ok(StreamEvent::MessageStop));
        Ok(Box::pin(stream::iter(events)))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        Ok(vec![ModelInfo {
            id: ModelId::new("mock-model"),
            provider_id: ProviderId::new("mock"),
            name: "Mock Model".to_string(),
            context_window: 200_000,
            max_output_tokens: 8_192,
        }])
    }

    async fn health_check(&self) -> Result<ProviderStatus, ProviderError> {
        Ok(ProviderStatus::Healthy)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            tool_calling: true,
            thinking: false,
            image_input: false,
            pdf_input: false,
            audio_input: false,
            video_input: false,
            caching: false,
            structured_output: false,
            system_prompt_style: SystemPromptStyle::SystemMessage,
        }
    }
}
