//! Read-only probes of local LLM servers. Only loopback, only GET, never inference.
use crate::traits::{HttpGet, LocalUrl};
use regex::Regex;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;

pub const PROBE_TIMEOUT: Duration = Duration::from_millis(500);

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ServerModel {
    pub name: Option<String>,
    pub r#type: Option<String>,
    pub quant: Option<String>,
    pub ctx: Option<i64>,
    pub size: Option<u64>,
    pub size_vram: Option<u64>,
    pub path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ServerInfo {
    pub kind: String,
    pub url: String,
    pub n_ctx: Option<i64>,
    pub slots_total: Option<i64>,
    pub slots_busy: Option<i64>,
    pub gen_tok_s: Option<f64>,
    pub prompt_tok_s: Option<f64>,
    pub model_path: Option<String>,
    pub models: Vec<ServerModel>,
}

impl ServerInfo {
    fn new(kind: &str, url: String, models: Vec<ServerModel>) -> Self {
        Self {
            kind: kind.into(),
            url,
            n_ctx: None,
            slots_total: None,
            slots_busy: None,
            gen_tok_s: None,
            prompt_tok_s: None,
            model_path: None,
            models,
        }
    }
}

enum Reply {
    Json(Value),
    Text(String),
}

fn fetch(http: &dyn HttpGet, port: u16, path: &str) -> Option<Reply> {
    let body = http.get(&LocalUrl::loopback(port, path), PROBE_TIMEOUT)?;
    Some(match serde_json::from_str::<Value>(&body) {
        Ok(v) => Reply::Json(v),
        Err(_) => Reply::Text(body),
    })
}

fn fetch_json(http: &dyn HttpGet, port: u16, path: &str) -> Option<Value> {
    match fetch(http, port, path)? {
        Reply::Json(v) => Some(v),
        Reply::Text(_) => None,
    }
}

fn metric(text: &str, name: &str) -> Option<f64> {
    let re = Regex::new(&format!(
        r"(?m)^llamacpp:{}\s+([\d.eE+-]+)",
        regex::escape(name)
    ))
    .ok()?;
    re.captures(text)?[1].parse().ok()
}

fn s(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::as_str).map(str::to_string)
}

/// Python truthiness for the "or" fallbacks: absent, null, 0 and "" are false.
fn int_or(a: Option<&Value>, b: Option<&Value>) -> Option<i64> {
    let ok = |v: &&Value| v.as_i64().is_some_and(|n| n != 0);
    a.filter(ok).or(b.filter(ok)).and_then(Value::as_i64)
}

fn basename(p: &str) -> String {
    p.rsplit('/').next().unwrap_or(p).to_string()
}

fn modality(o: &str) -> String {
    match o {
        "text" => "llm",
        "decisions" => "decision",
        "embeddings" | "embedding" => "embedding",
        other => other,
    }
    .to_string()
}

pub fn llama_server_models(v1: &Value, model_path: Option<&str>) -> Vec<ServerModel> {
    let empty = Vec::new();
    let data = v1.get("data").and_then(Value::as_array).unwrap_or(&empty);
    data.iter()
        .map(|m| {
            let first = m
                .pointer("/architecture/output_modalities")
                .and_then(Value::as_array)
                .and_then(|a| a.first())
                .and_then(Value::as_str);
            let id = m
                .get("id")
                .and_then(Value::as_str)
                .filter(|x| !x.is_empty());
            let meta = m.get("meta");
            ServerModel {
                name: Some(basename(
                    id.or(model_path).filter(|x| !x.is_empty()).unwrap_or("?"),
                )),
                r#type: first.map(modality),
                quant: meta.and_then(|x| s(x, "ftype")),
                ctx: meta.and_then(|x| x.get("n_ctx")).and_then(Value::as_i64),
                size: meta.and_then(|x| x.get("size")).and_then(Value::as_u64),
                size_vram: None,
                path: model_path.map(str::to_string),
            }
        })
        .collect()
}

pub fn ollama_models(ps: &Value) -> Vec<ServerModel> {
    let empty = Vec::new();
    ps.get("models")
        .and_then(Value::as_array)
        .unwrap_or(&empty)
        .iter()
        .map(|m| ServerModel {
            name: s(m, "name"),
            r#type: Some("llm".into()),
            quant: m.get("details").and_then(|d| s(d, "quantization_level")),
            ctx: m.get("context_length").and_then(Value::as_i64),
            size: m.get("size").and_then(Value::as_u64),
            size_vram: m.get("size_vram").and_then(Value::as_u64),
            path: None,
        })
        .collect()
}

/// Loaded models from LM Studio's `/api/v1/models` (preferred) or `/api/v0/models`.
pub fn lmstudio_models(v1: Option<&Value>, v0: Option<&Value>) -> Option<Vec<ServerModel>> {
    if let Some(models) = v1.and_then(|v| v.get("models")).and_then(Value::as_array) {
        let mut out = Vec::new();
        for m in models {
            let Some(insts) = m.get("loaded_instances").and_then(Value::as_array) else {
                continue;
            };
            for inst in insts {
                let mut t = s(m, "type");
                if t.as_deref() == Some("decision")
                    && let Some(d) = s(m, "decision_type").filter(|d| !d.is_empty())
                {
                    t = Some(format!("decision ({d})"));
                }
                let id = s(inst, "id")
                    .filter(|x| !x.is_empty())
                    .or_else(|| s(m, "key"));
                out.push(ServerModel {
                    name: id,
                    r#type: t,
                    quant: m.get("quantization").and_then(|q| s(q, "name")),
                    ctx: inst
                        .pointer("/config/context_length")
                        .and_then(Value::as_i64),
                    size: m.get("size_bytes").and_then(Value::as_u64),
                    size_vram: None,
                    path: None,
                });
            }
        }
        return Some(out);
    }
    let data = v0?.get("data")?.as_array()?;
    Some(
        data.iter()
            .filter(|m| m.get("state").and_then(Value::as_str) == Some("loaded"))
            .map(|m| ServerModel {
                name: s(m, "id"),
                r#type: s(m, "type"),
                quant: s(m, "quantization"),
                ctx: int_or(m.get("loaded_context_length"), m.get("max_context_length")),
                ..ServerModel::default()
            })
            .collect(),
    )
}

fn truthy(v: Option<&Value>) -> bool {
    match v {
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(Value::Null) | None => false,
        Some(Value::String(t)) => !t.is_empty(),
        Some(_) => true,
    }
}

/// Read-only GETs. Never calls inference, load or download endpoints.
pub fn probe_server(
    http: &dyn HttpGet,
    port: u16,
    prev: Option<&ServerInfo>,
) -> Option<ServerInfo> {
    let base = LocalUrl::loopback(port, "/").base();
    let props = fetch_json(http, port, "/props");
    if let Some(p) = props.as_ref().filter(|p| {
        p.is_object()
            && (p.get("default_generation_settings").is_some() || p.get("model_path").is_some())
    }) {
        let mut info = ServerInfo::new("llama-server", base, Vec::new());
        info.model_path = s(p, "model_path");
        info.n_ctx = int_or(
            p.pointer("/default_generation_settings/n_ctx"),
            p.get("n_ctx"),
        );
        info.slots_total = p.get("total_slots").and_then(Value::as_i64);
        info.models = match prev {
            Some(pr) if pr.kind == "llama-server" && pr.model_path == info.model_path => {
                pr.models.clone()
            }
            _ => match fetch_json(http, port, "/v1/models") {
                Some(v1) if v1.is_object() => llama_server_models(&v1, info.model_path.as_deref()),
                _ => Vec::new(),
            },
        };
        if let Some(Value::Array(slots)) = fetch_json(http, port, "/slots") {
            info.slots_total = Some(slots.len() as i64);
            info.slots_busy = Some(
                slots
                    .iter()
                    .filter(|x| truthy(x.get("is_processing")))
                    .count() as i64,
            );
        }
        if let Some(Reply::Text(m)) = fetch(http, port, "/metrics") {
            info.gen_tok_s = metric(&m, "predicted_tokens_seconds");
            info.prompt_tok_s = metric(&m, "prompt_tokens_seconds");
        }
        return Some(info);
    }
    if let Some(ps) =
        fetch_json(http, port, "/api/ps").filter(|p| p.get("models").is_some_and(Value::is_array))
    {
        return Some(ServerInfo::new("ollama", base, ollama_models(&ps)));
    }
    // Older LM Studio answers unknown endpoints with HTTP 200 and {"error": ...}, so fall back on shape, not status.
    let v1 = fetch_json(http, port, "/api/v1/models");
    let loaded = match lmstudio_models(v1.as_ref(), None) {
        Some(l) => Some(l),
        None => lmstudio_models(None, fetch_json(http, port, "/api/v0/models").as_ref()),
    };
    loaded.map(|l| ServerInfo::new("lmstudio", base, l))
}

/// Throttled, cached probes. Only localhost, only read-only GET endpoints.
#[derive(Default)]
pub struct ServerProber {
    cache: HashMap<u16, (f64, Option<ServerInfo>)>,
    last: HashMap<u16, ServerInfo>,
}

impl ServerProber {
    pub const TTL_HIT: f64 = 3.0;
    pub const TTL_MISS: f64 = 30.0;

    pub fn get(&mut self, http: &dyn HttpGet, ports: &[u16], now: f64) -> Option<ServerInfo> {
        for &port in ports {
            let fresh = self.cache.get(&port).and_then(|(t, info)| {
                let ttl = if info.is_some() {
                    Self::TTL_HIT
                } else {
                    Self::TTL_MISS
                };
                (now - t < ttl).then(|| info.clone())
            });
            let info = match fresh {
                Some(info) => info,
                None => {
                    let info = probe_server(http, port, self.last.get(&port));
                    self.cache.insert(port, (now, info.clone()));
                    if let Some(i) = &info {
                        self.last.insert(port, i.clone());
                    }
                    info
                }
            };
            if info.is_some() {
                return info;
            }
        }
        None
    }
}
