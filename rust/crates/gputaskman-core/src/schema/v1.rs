//! Reads a Python `--json` document (no `schema_version`) into the v2 model.
//!
//! Every missing key is `null` and every missing list is empty, as Python's `diff.py` reads optional
//! keys with `.get`. A counter that v1 reports as `null` becomes `unavailable(v1_unknown)`, never zero.
use std::collections::BTreeMap;

use serde_json::{Map, Value};

use super::SCHEMA_VERSION;
use super::v2::{
    AdapterV2, EnginesV2, LlmModelV2, LlmV2, MemoryV2, ModelFileV2, ProcessV2, ResidentV2,
    SegmentV2, SnapshotV2, StatusV2,
};
use crate::collector::Origin;
use crate::handy::HandyEvidence;
use crate::llm::servers::ServerInfo;
use crate::llm::view::SegmentKind;
use crate::model::{BackendKind, DeviceTelemetry, Evidence};
use crate::provenance::{Bytes, Confidence, EstimateSource, Reason, Reporter};

fn text(v: &Value, key: &str) -> Option<String> {
    v.get(key)?.as_str().map(String::from)
}

/// A non-empty string, as Python's `x or None`.
fn nonempty(v: &Value, key: &str) -> Option<String> {
    text(v, key).filter(|s| !s.is_empty())
}

fn uint(v: &Value, key: &str) -> Option<u64> {
    let n = v.get(key)?;
    n.as_u64().or_else(|| {
        n.as_f64()
            .filter(|f| f.is_finite() && *f >= 0.0)
            .map(|f| f as u64)
    })
}

fn float(v: &Value, key: &str) -> Option<f64> {
    v.get(key)?.as_f64()
}

fn items<'a>(v: &'a Value, key: &str) -> &'a [Value] {
    v.get(key)
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

fn measured_or_unknown(v: &Value, key: &str) -> Bytes {
    uint(v, key).map_or(Bytes::unavailable(Reason::V1Unknown), Bytes::measured)
}

fn confidence(s: Option<String>) -> Option<Confidence> {
    match s.as_deref() {
        Some("consistent") => Some(Confidence::Consistent),
        Some("order") => Some(Confidence::Order),
        _ => None,
    }
}

/// The amdgpu codes Python writes in `vram_unattributed_reason`.
fn drm_reason(code: Option<String>) -> Reason {
    match code.as_deref() {
        Some("legacy") => Reason::Legacy,
        Some("shared_client") => Reason::SharedClient,
        Some("shared_buffers") => Reason::SharedBuffers,
        Some("primary") => Reason::Primary,
        Some("incomplete") => Reason::Incomplete,
        Some("partial_scan") => Reason::PartialScan,
        _ => Reason::V1Unknown,
    }
}

fn evidence(v: &Value) -> Vec<Evidence> {
    let mut out: Vec<Evidence> = v
        .as_array()
        .map_or(&[][..], Vec::as_slice)
        .iter()
        .filter_map(|s| match s.as_str()? {
            "api" => Some(Evidence::Api),
            "mapped" => Some(Evidence::Mapped),
            "cmdline" => Some(Evidence::Cmdline),
            "installed" => Some(Evidence::Installed),
            _ => None,
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

fn model_file(m: &Value) -> ModelFileV2 {
    ModelFileV2 {
        path: text(m, "path").unwrap_or_default(),
        size: uint(m, "size").unwrap_or(0),
        resident: uint(m, "resident").unwrap_or(0),
        gguf: m.get("gguf").and_then(Value::as_bool).unwrap_or(false),
        evidence: evidence(m.get("sources").unwrap_or(&Value::Null)),
        store_name: text(m, "store_name"),
        kind: text(m, "kind").unwrap_or_default(),
        kind_hint: text(m, "kind_hint").unwrap_or_default(),
    }
}

fn segment_kind(s: Option<String>) -> SegmentKind {
    match s.as_deref() {
        Some("weights") => SegmentKind::Weights,
        Some("kv") => SegmentKind::Kv,
        Some("compute") => SegmentKind::Compute,
        Some("runtime") => SegmentKind::Runtime,
        _ => SegmentKind::Unknown,
    }
}

fn estimated(bytes: u64) -> Bytes {
    Bytes::Estimated {
        bytes,
        from: EstimateSource::Gguf,
    }
}

fn llm_model(g: &Value) -> LlmModelV2 {
    let file = model_file(g);
    LlmModelV2 {
        path: file.path,
        name: text(g, "name").unwrap_or_default(),
        label: text(g, "label").unwrap_or_default(),
        arch: text(g, "arch").unwrap_or_default(),
        kind: file.kind,
        kind_label: text(g, "kind_label").unwrap_or_default(),
        kind_hint: file.kind_hint,
        decision_type: text(g, "decision_type"),
        params: uint(g, "params"),
        quant: text(g, "quant").unwrap_or_default(),
        layers: uint(g, "layers")
            .and_then(|n| u32::try_from(n).ok())
            .unwrap_or(0),
        ctx_train: uint(g, "ctx_train"),
        gpu_weight: uint(g, "gpu_weight_bytes")
            .map_or(Bytes::unavailable(Reason::V1Unknown), estimated),
        evidence: evidence(g.get("sources").unwrap_or(&Value::Null)),
        store_name: file.store_name,
    }
}

fn server(v: &Value) -> Option<ServerInfo> {
    let mut o: Map<String, Value> = v.as_object()?.clone();
    if !o.get("models").is_some_and(Value::is_array) {
        o.insert("models".into(), Value::Array(vec![]));
    }
    serde_json::from_value(Value::Object(o)).ok()
}

fn llm(l: &Value) -> Option<LlmV2> {
    if l.as_object().is_none_or(Map::is_empty) {
        return None;
    }
    let segments = items(l, "segments")
        .iter()
        .map(|s| {
            let bytes = uint(s, "bytes");
            let exact = s.get("exact").and_then(Value::as_bool).unwrap_or(false);
            SegmentV2 {
                label: text(s, "label").unwrap_or_default(),
                kind: segment_kind(nonempty(s, "kind")),
                bytes: match (bytes, exact) {
                    (None, _) => Bytes::unavailable(Reason::V1Unknown),
                    (Some(bytes), true) => Bytes::Reported {
                        bytes,
                        by: Reporter::LoadLog,
                    },
                    (Some(bytes), false) => estimated(bytes),
                },
            }
        })
        .collect();
    Some(LlmV2 {
        models: items(l, "models").iter().map(llm_model).collect(),
        segments,
        kv_per_token: uint(l, "kv_bytes_per_token").map(estimated),
        n_ctx: l.get("n_ctx").and_then(Value::as_i64),
        flags: l
            .get("flags")
            .filter(|f| f.is_object())
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new())),
        server: l.get("server").and_then(server),
        log_source: text(l, "log_source"),
        notes: items(l, "notes")
            .iter()
            .filter_map(|n| n.as_str().map(String::from))
            .collect(),
    })
}

fn engines(p: &Value) -> EnginesV2 {
    let Some(busiest) = float(p, "gpu_pct") else {
        return EnginesV2::Unavailable {
            reason: Reason::V1Unknown,
        };
    };
    let by_engine: BTreeMap<String, f64> = p
        .get("engines")
        .and_then(Value::as_object)
        .map(|o| {
            o.iter()
                .filter_map(|(k, v)| Some((k.clone(), v.as_f64()?)))
                .collect()
        })
        .unwrap_or_default();
    EnginesV2::Measured { busiest, by_engine }
}

/// The same rule `from_snapshot` uses: outside rows cannot be read, local rows on a DRM backend carry
/// the amdgpu counters, and every other row has none.
fn resident(p: &Value, origin: Origin, backend: Option<BackendKind>) -> Option<ResidentV2> {
    match origin {
        Origin::Outside => Some(ResidentV2 {
            vram: Bytes::unavailable(Reason::OutsideNamespace),
            gtt: Bytes::unavailable(Reason::OutsideNamespace),
            summable: false,
        }),
        Origin::Local if backend == Some(BackendKind::Drm) => {
            let missing = if text(p, "vram_unattributed_reason").as_deref() == Some("incomplete") {
                Reason::Incomplete
            } else {
                Reason::Legacy
            };
            let get = |k| uint(p, k).map_or(Bytes::unavailable(missing), Bytes::measured);
            Some(ResidentV2 {
                vram: get("vram_resident_bytes"),
                gtt: get("gtt_resident_bytes"),
                summable: false,
            })
        }
        Origin::Local | Origin::Windows => None,
    }
}

fn process(p: &Value, backend: Option<BackendKind>) -> ProcessV2 {
    let by = confidence(nonempty(p, "vram_inferred_by"));
    let (vram, owner_inferred_by) = if let Some(b) = uint(p, "vram_bytes") {
        (Bytes::measured(b), by)
    } else if let Some(bytes) = uint(p, "vram_inferred_bytes") {
        // Python always names the inference; if it is missing, claim the weaker one.
        let by = by.unwrap_or(Confidence::Order);
        (Bytes::Inferred { bytes, by }, None)
    } else {
        (
            Bytes::unavailable(drm_reason(nonempty(p, "vram_unattributed_reason"))),
            None,
        )
    };
    let origin = match text(p, "host").as_deref() {
        Some("outside") => Origin::Outside,
        Some("windows") => Origin::Windows,
        _ => Origin::Local,
    };
    ProcessV2 {
        pid: uint(p, "pid").and_then(|n| u32::try_from(n).ok()),
        origin,
        name: text(p, "name").unwrap_or_default(),
        kind: text(p, "kind").unwrap_or_default(),
        runtimes: items(p, "runtimes")
            .iter()
            .filter_map(|r| r.as_str().map(String::from))
            .collect(),
        vram,
        owner_inferred_by,
        shared: measured_or_unknown(p, "shared_bytes"),
        committed: measured_or_unknown(p, "committed_bytes"),
        paged_out: measured_or_unknown(p, "paged_out_bytes"),
        resident: resident(p, origin, backend),
        engines: engines(p),
        ports: items(p, "ports")
            .iter()
            .filter_map(|n| u16::try_from(n.as_u64()?).ok())
            .collect(),
        inside: text(p, "inside").unwrap_or_default(),
        model_files: items(p, "model_files").iter().map(model_file).collect(),
        installed_models: items(p, "installed_models")
            .iter()
            .map(model_file)
            .collect(),
        handy: p
            .get("handy")
            .filter(|h| h.is_object())
            .and_then(|h| serde_json::from_value::<HandyEvidence>(h.clone()).ok()),
        llm: p.get("llm").and_then(llm),
    }
}

fn adapter(v: &Value, name: &str) -> AdapterV2 {
    let a = v.get("adapter").filter(|a| a.is_object());
    let get = |k| a.and_then(|a| uint(a, k));
    AdapterV2 {
        name: name.to_string(),
        vendor: a.and_then(|a| text(a, "vendor")).unwrap_or_default(),
        vendor_id: get("vendor_id")
            .and_then(|n| u16::try_from(n).ok())
            .unwrap_or(0),
        key: a.and_then(|a| text(a, "key")).unwrap_or_default(),
        dedicated_vram: get("dedicated_vram").unwrap_or(0),
        shared_system: get("shared_system").unwrap_or(0),
        integrated: a
            .and_then(|a| a.get("integrated"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
    }
}

/// Overlay the keys v1 has onto the telemetry defaults; `None` when v1 has no device.
fn device(v: &Value) -> Option<DeviceTelemetry> {
    let src = v.as_object()?;
    let Value::Object(mut base) = serde_json::to_value(DeviceTelemetry::default()).ok()? else {
        return None;
    };
    for (k, val) in src {
        if !val.is_null() && base.contains_key(k) {
            base.insert(k.clone(), val.clone());
        }
    }
    serde_json::from_value(Value::Object(base)).ok()
}

/// An unrecognised backend name becomes `None`, with a note.
fn backend(v: &Value) -> Option<BackendKind> {
    match text(v, "backend").as_deref() {
        Some("nvml") => Some(BackendKind::Nvml),
        Some("drm") => Some(BackendKind::Drm),
        Some("wddm") => Some(BackendKind::Wddm),
        Some("wsl") => Some(BackendKind::Wsl),
        Some("metal") => Some(BackendKind::Metal),
        _ => None,
    }
}

pub fn from_v1(v: &Value) -> Result<SnapshotV2, String> {
    if !v.is_object() {
        return Err("a v1 snapshot must be a JSON object".to_string());
    }
    let mut notes = Vec::new();
    if let Some(b) = text(v, "backend")
        && backend(v).is_none()
    {
        notes.push(format!(
            "v1 backend '{b}' is not recognised; the backend is unknown"
        ));
    }
    Ok(SnapshotV2 {
        schema_version: SCHEMA_VERSION,
        taken: float(v, "taken").filter(|t| t.is_finite()),
        backend: backend(v),
        adapter: nonempty(v, "gpu").map(|name| adapter(v, &name)),
        driver: nonempty(v, "driver"),
        cuda: nonempty(v, "cuda"),
        memory: MemoryV2 {
            total: measured_or_unknown(v, "vram_total"),
            used: measured_or_unknown(v, "vram_used"),
            shared_used: measured_or_unknown(v, "shared_used"),
            attributed: measured_or_unknown(v, "attributed"),
            shared_is_spill: v
                .get("shared_is_spill")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        },
        status: StatusV2 {
            process: nonempty(v, "process_status"),
            device: None,
            counter: nonempty(v, "counter_status"),
            host: nonempty(v, "host_status"),
            wsl_split: nonempty(v, "wsl_split"),
        },
        device: v.get("device").and_then(device),
        notes,
        processes: items(v, "processes")
            .iter()
            .map(|p| process(p, backend(v)))
            .collect(),
    })
}
