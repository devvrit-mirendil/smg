//! HTTP middleware that logs non-prompt request parameters at the HTTP layer.
//!
//! Reads the request body, strips large/sensitive fields (messages, prompt,
//! input, tools descriptions/schemas), logs the remainder at `info!` level
//! with target `smg::request_params`, then puts the body back so downstream
//! handlers receive it unchanged.

use std::{
    pin::Pin,
    task::{Context, Poll},
};

use axum::{body::Body, extract::Request, response::Response};
use bytes::Bytes;
use http::Method;
use serde_json::Value;
use tower::{Layer, Service};
use tracing::info;

/// Fields removed entirely before logging (contain prompt/message content).
const BLACKLIST: &[&str] = &[
    "messages",
    "prompt",
    "input",
    "input_ids",
    "functions",
    "logit_bias",
    "embedding_bias",
];

/// Maximum body size to buffer (4 MB). Larger bodies are skipped silently.
const MAX_BODY_BYTES: usize = 4 * 1024 * 1024;

/// Tower Layer that enables request param logging.
#[derive(Clone)]
pub struct RequestParamLogLayer {
    enabled: bool,
}

impl RequestParamLogLayer {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }
}

impl<S> Layer<S> for RequestParamLogLayer {
    type Service = RequestParamLogMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RequestParamLogMiddleware {
            inner,
            enabled: self.enabled,
        }
    }
}

/// Tower Service that performs the actual body read + filter + log + restore.
#[derive(Clone)]
pub struct RequestParamLogMiddleware<S> {
    inner: S,
    enabled: bool,
}

impl<S> Service<Request> for RequestParamLogMiddleware<S>
where
    S: Service<Request, Response = Response> + Send + Clone + 'static,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future =
        Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request) -> Self::Future {
        if !self.enabled {
            return Box::pin(self.inner.call(req));
        }

        let method = req.method().clone();
        let path = req.uri().path().to_owned();
        let mut inner = self.inner.clone();

        Box::pin(async move {
            let (parts, body) = req.into_parts();

            // Collect body bytes, falling back transparently on error
            let body_bytes = match axum::body::to_bytes(body, MAX_BODY_BYTES).await {
                Ok(bytes) => bytes,
                Err(_) => {
                    // Body too large or read error — pass through with empty body
                    let req = Request::from_parts(parts, Body::empty());
                    return inner.call(req).await;
                }
            };

            log_filtered_params(&method, &path, &body_bytes);

            // Restore the body for downstream handlers
            let req = Request::from_parts(parts, Body::from(body_bytes));
            inner.call(req).await
        })
    }
}

fn log_filtered_params(method: &Method, path: &str, body: &Bytes) {
    let Ok(text) = std::str::from_utf8(body) else {
        return;
    };

    // Only log JSON object bodies (all inference endpoints use these)
    let Ok(mut value) = serde_json::from_str::<Value>(text) else {
        return;
    };

    let Some(obj) = value.as_object_mut() else {
        return;
    };

    // Remove blacklisted fields
    for key in BLACKLIST {
        obj.remove(*key);
    }

    // Keep tools but strip per-tool description/parameters/input_schema
    if let Some(Value::Array(tools)) = obj.get_mut("tools") {
        for tool in tools.iter_mut() {
            if let Some(func) = tool
                .as_object_mut()
                .and_then(|t| t.get_mut("function"))
                .and_then(|f| f.as_object_mut())
            {
                func.remove("description");
                func.remove("parameters");
                func.remove("input_schema");
            }
        }
    }

    info!(
        target: "smg::request_params",
        method = %method,
        path = %path,
        params = %value,
    );
}
