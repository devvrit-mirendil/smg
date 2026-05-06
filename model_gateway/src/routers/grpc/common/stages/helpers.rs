//! Common helper functions shared across stages

use std::sync::Arc;

use openai_protocol::chat::ChatMessage;
use rand::Rng;
use sha2::{Digest, Sha256};
use smg_grpc_client::sglang_proto::DisaggregatedParams;
use tracing::{debug, info};

use crate::{
    routers::grpc::{context::WorkerSelection, proto_wrapper::ProtoGenerateRequest},
    worker::{RuntimeType, Worker, DEFAULT_BOOTSTRAP_PORT},
};

/// Inject PD bootstrap metadata for SGLang if needed.
///
/// SGLang uses DisaggregatedParams with bootstrap host/port/room.
/// vLLM uses different mechanisms: NIXL (automatic prefix matching) or
/// Mooncake (kv_transfer_params injected in request_execution stage).
pub(crate) fn maybe_inject_pd_metadata(
    request: &mut ProtoGenerateRequest,
    workers: &WorkerSelection,
) {
    if let WorkerSelection::Dual {
        prefill,
        runtime_type,
        ..
    } = workers
    {
        if *runtime_type == RuntimeType::Sglang {
            inject_sglang_bootstrap_metadata(request, prefill);
        }
    }
}

/// Inject bootstrap metadata into a SGLang gRPC request.
fn inject_sglang_bootstrap_metadata(
    request: &mut ProtoGenerateRequest,
    prefill_worker: &Arc<dyn Worker>,
) {
    let metadata = prefill_worker.metadata();
    let hostname = metadata.bootstrap_host();
    let bootstrap_port = metadata.bootstrap_port().unwrap_or(DEFAULT_BOOTSTRAP_PORT);
    let room_id = rand::rng().random_range(0..i32::MAX);

    let disagg_params = DisaggregatedParams {
        bootstrap_host: hostname.to_string(),
        bootstrap_port: bootstrap_port as i32,
        bootstrap_room: room_id,
    };

    let sglang_request = request.as_sglang_mut();
    sglang_request.disaggregated_params = Some(disagg_params);

    debug!(
        "Injected bootstrap metadata: host={}, port={}, room={}",
        hostname, bootstrap_port, room_id
    );
}

fn chat_message_role(msg: &ChatMessage) -> &'static str {
    match msg {
        ChatMessage::System { .. } => "system",
        ChatMessage::User { .. } => "user",
        ChatMessage::Assistant { .. } => "assistant",
        ChatMessage::Tool { .. } => "tool",
        ChatMessage::Function { .. } => "function",
        ChatMessage::Developer { .. } => "developer",
    }
}

fn chat_message_text_content(msg: &ChatMessage) -> String {
    match msg {
        ChatMessage::System { content, .. }
        | ChatMessage::User { content, .. }
        | ChatMessage::Tool { content, .. }
        | ChatMessage::Developer { content, .. } => content.to_simple_string(),
        ChatMessage::Assistant { content, .. } => content
            .as_ref()
            .map_or_else(String::new, |c| c.to_simple_string()),
        ChatMessage::Function { content, .. } => content.clone(),
    }
}

/// Compute per-message SHA-256 hashes matching TRT-LLM's `openai_server.py` format:
/// `sha256(role + "\x00" + content).hexdigest()[:12]`
pub(crate) fn compute_and_log_message_hashes(
    request_id: &str,
    messages: &[ChatMessage],
) -> Vec<(String, String)> {
    let hashes: Vec<(String, String)> = messages
        .iter()
        .map(|msg| {
            let role = chat_message_role(msg);
            let content = chat_message_text_content(msg);
            let mut hasher = Sha256::new();
            hasher.update(format!("{role}\x00{content}").as_bytes());
            let hash = format!("{:x}", hasher.finalize());
            (role.to_string(), hash[..12].to_string())
        })
        .collect();
    debug!(
        target: "smg::request",
        request_id = %request_id,
        message_hashes = ?hashes,
        "Request message hashes for session reconstruction"
    );
    hashes
}

/// Log non-prompt sampling parameters from a ChatCompletionRequest at INFO level.
pub(crate) fn log_chat_request_params(
    request_id: &str,
    request: &openai_protocol::chat::ChatCompletionRequest,
) {
    use openai_protocol::{
        common::{ResponseFormat, ToolChoice, ToolChoiceValue},
        messages::ThinkingConfig,
    };

    let response_format = request.response_format.as_ref().map(|rf| match rf {
        ResponseFormat::Text => "text".to_string(),
        ResponseFormat::JsonObject => "json_object".to_string(),
        ResponseFormat::JsonSchema { json_schema } => {
            format!("json_schema({})", json_schema.name)
        }
    });

    let tool_choice = request.tool_choice.as_ref().map(|tc| match tc {
        ToolChoice::Value(ToolChoiceValue::Auto) => "auto".to_string(),
        ToolChoice::Value(ToolChoiceValue::Required) => "required".to_string(),
        ToolChoice::Value(ToolChoiceValue::None) => "none".to_string(),
        ToolChoice::Function { function, .. } => format!("function({})", function.name),
        ToolChoice::AllowedTools { mode, tools, .. } => {
            let names: Vec<_> = tools
                .iter()
                .map(|t| match t {
                    openai_protocol::common::ToolReference::Function { name } => name.as_str(),
                    openai_protocol::common::ToolReference::Mcp { server_label, .. } => server_label.as_str(),
                    _ => "hosted_tool",
                })
                .collect();
            format!("allowed_tools:{mode} [{names}]", names = names.join(", "))
        }
    });

    let tool_names: Option<Vec<String>> = request.tools.as_ref().map(|tools| {
        tools.iter().map(|t| t.function.name.clone()).collect()
    });

    let thinking = request.thinking.as_ref().map(|t| match t {
        ThinkingConfig::Enabled { budget_tokens } => {
            format!("enabled(budget={})", budget_tokens)
        }
        ThinkingConfig::Disabled => "disabled".to_string(),
    });

    #[expect(
        deprecated,
        reason = "max_tokens is the legacy fallback for max_completion_tokens"
    )]
    let max_tokens = request.max_completion_tokens.or(request.max_tokens);
    #[expect(
        deprecated,
        reason = "seed is the legacy field; log it for completeness"
    )]
    let seed = request.seed;

    info!(
        target: "smg::request_params",
        request_id = %request_id,
        model = %request.model,
        temperature = ?request.temperature,
        top_p = ?request.top_p,
        top_k = ?request.top_k,
        min_p = ?request.min_p,
        max_tokens = ?max_tokens,
        frequency_penalty = ?request.frequency_penalty,
        presence_penalty = ?request.presence_penalty,
        repetition_penalty = ?request.repetition_penalty,
        logprobs = request.logprobs,
        top_logprobs = ?request.top_logprobs,
        n = ?request.n,
        stop = ?request.stop,
        stream = request.stream,
        response_format = ?response_format,
        tool_choice = ?tool_choice,
        tool_names = ?tool_names,
        seed = ?seed,
        reasoning_effort = ?request.reasoning_effort,
        thinking = ?thinking,
        "Request parameters"
    );
}

/// Log non-prompt sampling parameters from a CreateMessageRequest at INFO level.
pub(crate) fn log_messages_request_params(
    request_id: &str,
    request: &openai_protocol::messages::CreateMessageRequest,
) {
    use openai_protocol::messages::{ThinkingConfig, ToolChoice};

    let tool_choice = request.tool_choice.as_ref().map(|tc| match tc {
        ToolChoice::Auto { .. } => "auto".to_string(),
        ToolChoice::Any { .. } => "any".to_string(),
        ToolChoice::Tool { name, .. } => name.clone(),
        ToolChoice::None => "none".to_string(),
    });

    let tool_names: Option<Vec<String>> = request.tools.as_ref().map(|tools| {
        use openai_protocol::messages::Tool;
        tools
            .iter()
            .map(|t| match t {
                Tool::Custom(c) => c.name.clone(),
                Tool::McpToolset(_) => "mcp_toolset".to_string(),
                Tool::ToolSearch(_) => "tool_search".to_string(),
                Tool::Bash(_) => "bash".to_string(),
                Tool::TextEditor(_) => "text_editor".to_string(),
                Tool::WebSearch(_) => "web_search".to_string(),
            })
            .collect()
    });

    let thinking = request.thinking.as_ref().map(|t| match t {
        ThinkingConfig::Enabled { budget_tokens } => {
            format!("enabled(budget={})", budget_tokens)
        }
        ThinkingConfig::Disabled => "disabled".to_string(),
    });

    let stream = request.stream.unwrap_or(false);

    info!(
        target: "smg::request_params",
        request_id = %request_id,
        model = %request.model,
        temperature = ?request.temperature,
        top_p = ?request.top_p,
        top_k = ?request.top_k,
        max_tokens = request.max_tokens,
        stop_sequences = ?request.stop_sequences,
        stream = stream,
        tool_choice = ?tool_choice,
        tool_names = ?tool_names,
        thinking = ?thinking,
        "Request parameters"
    );
}

/// Log non-prompt sampling parameters from a CompletionRequest at INFO level.
pub(crate) fn log_completion_request_params(
    request_id: &str,
    request: &openai_protocol::completion::CompletionRequest,
) {
    info!(
        target: "smg::request_params",
        request_id = %request_id,
        model = %request.model,
        temperature = ?request.temperature,
        top_p = ?request.top_p,
        top_k = ?request.top_k,
        min_p = ?request.min_p,
        max_tokens = ?request.max_tokens,
        frequency_penalty = ?request.frequency_penalty,
        presence_penalty = ?request.presence_penalty,
        repetition_penalty = ?request.repetition_penalty,
        logprobs = ?request.logprobs,
        n = ?request.n,
        stop = ?request.stop,
        stream = request.stream,
        seed = ?request.seed,
        "Request parameters"
    );
}

/// Compute per-message SHA-256 hashes from InputMessage (Messages API) format.
pub(crate) fn compute_and_log_input_message_hashes(
    request_id: &str,
    messages: &[openai_protocol::messages::InputMessage],
) -> Vec<(String, String)> {
    use openai_protocol::messages::Role;
    let hashes: Vec<(String, String)> = messages
        .iter()
        .map(|msg| {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            let content = match &msg.content {
                openai_protocol::messages::InputContent::String(s) => s.clone(),
                openai_protocol::messages::InputContent::Blocks(blocks) => blocks
                    .iter()
                    .filter_map(|b| {
                        if let openai_protocol::messages::InputContentBlock::Text(t) = b {
                            Some(t.text.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
            };
            let mut hasher = Sha256::new();
            hasher.update(format!("{role}\x00{content}").as_bytes());
            let hash = format!("{:x}", hasher.finalize());
            (role.to_string(), hash[..12].to_string())
        })
        .collect();
    debug!(
        target: "smg::request",
        request_id = %request_id,
        message_hashes = ?hashes,
        "Request message hashes for session reconstruction"
    );
    hashes
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use openai_protocol::chat::{ChatCompletionRequest, ChatMessage, MessageContent};
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    struct SharedBuffer(Arc<Mutex<Vec<u8>>>);
    impl std::io::Write for SharedBuffer {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn capture_log<F: FnOnce()>(f: F) -> String {
        let buf = Arc::new(Mutex::new(Vec::<u8>::new()));
        let buf_clone = buf.clone();
        let make_writer = move || SharedBuffer(buf_clone.clone());
        let subscriber = tracing_subscriber::registry()
            .with(EnvFilter::new("smg::request_params=info"))
            .with(fmt::layer().with_writer(make_writer).with_ansi(false));
        tracing::subscriber::with_default(subscriber, f);
        let data = buf.lock().unwrap().clone();
        String::from_utf8(data).unwrap_or_default()
    }

    fn make_chat_request(model: &str, messages_content: &str) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: model.to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text(messages_content.to_string()),
                name: None,
            }],
            temperature: Some(0.7),
            top_p: Some(0.9),
            max_completion_tokens: Some(100),
            stream: false,
            ..Default::default()
        }
    }

    #[test]
    fn test_log_chat_request_params_contains_expected_fields() {
        let req = make_chat_request("test-model", "hello world secret content");
        let output = capture_log(|| super::log_chat_request_params("req-123", &req));

        assert!(output.contains("smg::request_params"), "log target missing");
        assert!(output.contains("req-123"), "request_id missing");
        assert!(output.contains("test-model"), "model missing");
        assert!(output.contains("0.7"), "temperature missing");
        assert!(output.contains("0.9"), "top_p missing");
        assert!(output.contains("100"), "max_tokens missing");
        assert!(output.contains("stream=false"), "stream missing");
    }

    #[test]
    fn test_log_chat_request_params_no_content_leakage() {
        let secret = "UNIQUESECRETCONTENT12345";
        let req = make_chat_request("test-model", secret);
        let output = capture_log(|| super::log_chat_request_params("req-456", &req));

        assert!(!output.contains(secret), "message content leaked into log");
    }

    #[test]
    fn test_log_chat_request_params_tool_choice_and_count() {
        use openai_protocol::common::{Function, Tool, ToolChoice, ToolChoiceValue};
        let mut req = make_chat_request("test-model", "hello");
        req.tools = Some(vec![
            Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: "get_weather".to_string(),
                    description: Some("Get current weather".to_string()),
                    parameters: serde_json::Value::Null,
                    strict: None,
                },
            },
            Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: "search".to_string(),
                    description: None,
                    parameters: serde_json::Value::Null,
                    strict: None,
                },
            },
        ]);
        req.tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Auto));

        let output = capture_log(|| super::log_chat_request_params("req-789", &req));

        assert!(output.contains("auto"), "tool_choice 'auto' missing");
        assert!(output.contains("tool_count=Some(2)"), "tool count missing");
        assert!(!output.contains("get_weather"), "tool definition leaked");
        assert!(
            !output.contains("Get current weather"),
            "tool description leaked"
        );
    }

    #[test]
    fn test_log_chat_request_params_response_format() {
        use openai_protocol::common::ResponseFormat;
        let mut req = make_chat_request("test-model", "hello");
        req.response_format = Some(ResponseFormat::JsonObject);

        let output = capture_log(|| super::log_chat_request_params("req-rf", &req));

        assert!(output.contains("json_object"), "response_format missing");
    }
}
