//! Session Tracing — OpenTelemetry span stubs.
//!
//! Telemetry spans are no-ops by default. When the `otel` feature is enabled,
//! spans are exported through OpenTelemetry.

use std::sync::Arc;

/// Shared span interface used by the query loop.
pub trait SpanLike: Send + Sync {
    fn set_attribute(&self, key: &str, value: &str);
    fn set_attributes(&self, attrs: &[(&str, &str)]);
    fn add_event(&self, name: &str);
    fn record_exception(&self, error: &str);
    fn end(&self);
}

/// A no-op span that implements the minimal span interface.
#[derive(Debug, Clone)]
pub struct NoopSpan;

impl NoopSpan {
    /// Create a new no-op span.
    pub fn new() -> Self {
        Self
    }

    /// Set a single attribute (no-op).
    pub fn set_attribute(&self, _key: &str, _value: &str) {}

    /// Set multiple attributes (no-op).
    pub fn set_attributes(&self, _attrs: &[(&str, &str)]) {}

    /// Add an event to the span (no-op).
    pub fn add_event(&self, _name: &str) {}

    /// Record an exception (no-op).
    pub fn record_exception(&self, _error: &str) {}

    /// End the span (no-op).
    pub fn end(&self) {}
}

impl Default for NoopSpan {
    fn default() -> Self {
        Self::new()
    }
}

impl SpanLike for NoopSpan {
    fn set_attribute(&self, _: &str, _: &str) {}
    fn set_attributes(&self, _: &[(&str, &str)]) {}
    fn add_event(&self, _: &str) {}
    fn record_exception(&self, _: &str) {}
    fn end(&self) {}
}

#[cfg(feature = "otel")]
pub(crate) mod otel_impl {
    use super::SpanLike;
    use opentelemetry::global;
    use opentelemetry::trace::{Span as _, Status, Tracer as _};
    use opentelemetry::KeyValue;
    use opentelemetry_otlp::WithExportConfig;
    use opentelemetry_sdk::{runtime::Tokio, trace::Config as TraceConfig, Resource};
    use opentelemetry_sdk::trace::TracerProvider as SdkProvider;
    use parking_lot::Mutex;
    use std::sync::OnceLock;

    pub static PROVIDER: OnceLock<SdkProvider> = OnceLock::new();

    pub fn init_provider() {
        if PROVIDER.get().is_some() {
            return;
        }

        let endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
            .unwrap_or_else(|_| "http://localhost:4317".to_string());

        let trace_config = TraceConfig::default()
            .with_resource(Resource::new(vec![KeyValue::new("service.name", "mangocode")]));

        let provider = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_trace_config(trace_config)
            .with_exporter(opentelemetry_otlp::new_exporter().tonic().with_endpoint(&endpoint))
            .install_batch(Tokio)
            .expect("Failed to install OTLP trace pipeline");

        let _ = global::set_tracer_provider(provider.clone());
        let _ = PROVIDER.set(provider);
    }

    pub fn tracer() -> Option<global::BoxedTracer> {
        PROVIDER.get().map(|_| global::tracer("mangocode"))
    }

    #[derive(Debug)]
    pub struct OtelSpanWrapper {
        span: Mutex<global::BoxedSpan>,
    }

    impl OtelSpanWrapper {
        pub fn new(span: global::BoxedSpan) -> Self {
            Self {
                span: Mutex::new(span),
            }
        }
    }

    impl SpanLike for OtelSpanWrapper {
        fn set_attribute(&self, key: &str, value: &str) {
            self.span
                .lock()
                .set_attribute(KeyValue::new(key.to_string(), value.to_string()));
        }

        fn set_attributes(&self, attrs: &[(&str, &str)]) {
            let mut span = self.span.lock();
            span.set_attributes(attrs.iter().map(|(key, value)| {
                KeyValue::new((*key).to_string(), (*value).to_string())
            }));
        }

        fn add_event(&self, name: &str) {
            self.span.lock().add_event(name.to_string(), Vec::new());
        }

        fn record_exception(&self, error: &str) {
            let err = std::io::Error::other(error.to_string());
            let mut span = self.span.lock();
            span.record_error(&err);
            span.set_status(Status::error(error.to_string()));
        }

        fn end(&self) {
            self.span.lock().end();
        }
    }

    pub fn boxed_span(name: String) -> Option<std::sync::Arc<dyn SpanLike>> {
        let tracer = tracer()?;
        let span = tracer.start(name);
        Some(std::sync::Arc::new(OtelSpanWrapper::new(span)))
    }
}

fn new_span(name: &str) -> Arc<dyn SpanLike> {
    let _ = name;
    #[cfg(feature = "otel")]
    {
        if let Some(span) = otel_impl::boxed_span(name.to_string()) {
            return span;
        }
    }

    Arc::new(NoopSpan::new())
}

// ---------------------------------------------------------------------------
// Public Span API
// ---------------------------------------------------------------------------

/// Start an interaction span (root span for a user request).
pub fn start_interaction_span(request_id: &str) -> Arc<dyn SpanLike> {
    let span = new_span("interaction");
    span.set_attribute("request_id", request_id);
    span
}

/// End an interaction span.
pub fn end_interaction_span(span: Arc<dyn SpanLike>) {
    span.end();
}

/// Start an LLM request span (traces API calls).
pub fn start_llm_request_span(model: &str, max_tokens: u32) -> Arc<dyn SpanLike> {
    let span = new_span("llm.request");
    span.set_attribute("model", model);
    span.set_attribute("max_tokens", &max_tokens.to_string());
    span
}

/// End an LLM request span.
pub fn end_llm_request_span(span: Arc<dyn SpanLike>, input_tokens: u64, output_tokens: u64) {
    span.set_attribute("input_tokens", &input_tokens.to_string());
    span.set_attribute("output_tokens", &output_tokens.to_string());
    span.end();
}

/// Start a tool execution span.
pub fn start_tool_span(tool_name: &str) -> Arc<dyn SpanLike> {
    let span = new_span("tool.execute");
    span.set_attribute("tool_name", tool_name);
    span
}

/// End a tool execution span.
pub fn end_tool_span(span: Arc<dyn SpanLike>, success: bool, error: Option<&str>) {
    span.set_attribute("success", if success { "true" } else { "false" });
    if let Some(error) = error {
        span.record_exception(error);
        span.set_attribute("error", error);
    }
    span.end();
}

/// Start a permission dialog span.
pub fn start_permission_span(tool_name: &str) -> Arc<dyn SpanLike> {
    let span = new_span("permission.check");
    span.set_attribute("tool_name", tool_name);
    span
}

/// End a permission dialog span.
pub fn end_permission_span(span: Arc<dyn SpanLike>) {
    span.end();
}

/// Start a hook execution span.
pub fn start_hook_span(hook_name: &str) -> Arc<dyn SpanLike> {
    let span = new_span("hook.execute");
    span.set_attribute("hook_name", hook_name);
    span
}

/// End a hook execution span.
pub fn end_hook_span(span: Arc<dyn SpanLike>) {
    span.end();
}

/// Add tool content event to span.
pub fn add_tool_content_event(span: &Arc<dyn SpanLike>, label: &str, content: &str) {
    span.add_event(label);
    span.set_attribute("tool_content_label", label);
    span.set_attribute("tool_content", content);
}

/// Execute an async operation within a span (no-op wrapper).
pub async fn execute_in_span<F, T>(f: F) -> T
where
    F: std::future::Future<Output = T>,
{
    f.await
}

/// Check if enhanced telemetry is enabled.
pub fn is_enhanced_telemetry_enabled() -> bool {
    crate::analytics::is_telemetry_enabled()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_span_methods() {
        let span = NoopSpan::new();
        span.set_attribute("key", "value");
        span.set_attributes(&[("k1", "v1"), ("k2", "v2")]);
        span.add_event("test_event");
        span.record_exception("test error");
        span.end();
    }

    #[test]
    fn test_span_functions() {
        let root = start_interaction_span("req-123");
        end_interaction_span(root);

        let llm = start_llm_request_span("claude-3", 4096);
        end_llm_request_span(llm, 100, 50);

        let tool = start_tool_span("bash");
        end_tool_span(tool, true, None);

        let perm = start_permission_span("read_file");
        end_permission_span(perm);

        let hook = start_hook_span("pre_request");
        end_hook_span(hook);

        assert!(!is_enhanced_telemetry_enabled());
    }

    #[tokio::test]
    async fn test_execute_in_span_async() {
        let result = execute_in_span(async { 42 }).await;
        assert_eq!(result, 42);
    }
}