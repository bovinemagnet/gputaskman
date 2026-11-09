//! JSON schema v2: every number carries its provenance, unknown is `unavailable` with a reason and
//! never omitted or zero. Optional objects serialise as `null`.
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::SCHEMA_VERSION;
use crate::collector::{Origin, ProcRow, Snapshot};
use crate::handy::HandyEvidence;
use crate::llm::servers::ServerInfo;
use crate::llm::view::{LlmView, SegmentKind};
use crate::model::{Adapter, BackendKind, DeviceTelemetry, Engines, Evidence, MappedModel, Pid};
use crate::provenance::{Bytes, Confidence, Reason, Resident};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterV2 {
    pub name: String,
    pub vendor: String,
    pub vendor_id: u16,
    pub key: String,
    pub dedicated_vram: u64,
    pub shared_system: u64,
    pub integrated: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryV2 {
    pub total: Bytes,
    pub used: Bytes,
    pub shared_used: Bytes,
    pub attributed: Bytes,
    pub shared_is_spill: bool,
}

/// Why something could not be read. Counter, host and WSL split exist for the Windows port and are
/// always `null` on Linux.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatusV2 {
    pub process: Option<String>,
    pub device: Option<String>,
    pub counter: Option<String>,
    pub host: Option<String>,
    pub wsl_split: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum EnginesV2 {
    Measured {
        busiest: f64,
        by_engine: BTreeMap<String, f64>,
    },
    Unavailable {
        reason: Reason,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResidentV2 {
    pub vram: Bytes,
    pub gtt: Bytes,
    /// Always false: the same buffer can appear under several processes.
    pub summable: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelFileV2 {
    pub path: String,
    pub size: u64,
    pub resident: u64,
    pub gguf: bool,
    pub evidence: Vec<Evidence>,
    pub store_name: Option<String>,
    pub kind: String,
    pub kind_hint: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmModelV2 {
    pub path: String,
    pub name: String,
    pub label: String,
    pub arch: String,
    pub kind: String,
    pub kind_label: String,
    pub kind_hint: String,
    pub decision_type: Option<String>,
    pub params: Option<u64>,
    pub quant: String,
    pub layers: u32,
    pub ctx_train: Option<u64>,
    pub gpu_weight: Bytes,
    pub evidence: Vec<Evidence>,
    pub store_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentV2 {
    pub label: String,
    pub kind: SegmentKind,
    pub bytes: Bytes,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmV2 {
    pub models: Vec<LlmModelV2>,
    pub segments: Vec<SegmentV2>,
    pub kv_per_token: Option<Bytes>,
    pub n_ctx: Option<i64>,
    pub flags: serde_json::Value,
    pub server: Option<ServerInfo>,
    pub log_source: Option<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessV2 {
    /// `None` only when read from a v1 document that has no pid: unknown is not zero.
    pub pid: Option<Pid>,
    pub origin: Origin,
    pub name: String,
    pub kind: String,
    pub runtimes: Vec<String>,
    pub vram: Bytes,
    pub owner_inferred_by: Option<Confidence>,
    pub shared: Bytes,
    pub committed: Bytes,
    pub paged_out: Bytes,
    pub resident: Option<ResidentV2>,
    pub engines: EnginesV2,
    pub ports: Vec<u16>,
    pub inside: String,
    pub model_files: Vec<ModelFileV2>,
    pub installed_models: Vec<ModelFileV2>,
    pub handy: Option<HandyEvidence>,
    pub llm: Option<LlmV2>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapshotV2 {
    pub schema_version: u32,
    pub taken: f64,
    pub backend: Option<BackendKind>,
    pub adapter: Option<AdapterV2>,
    pub driver: Option<String>,
    pub cuda: Option<String>,
    pub memory: MemoryV2,
    pub status: StatusV2,
    pub device: Option<DeviceTelemetry>,
    /// Degradation reasons (a vendor library that did not load, say). Not in the spec's original
    /// top level; rule 9 needs them in the JSON as well as the text.
    pub notes: Vec<String>,
    pub processes: Vec<ProcessV2>,
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn adapter(a: &Adapter) -> AdapterV2 {
    AdapterV2 {
        name: a.name.clone(),
        vendor: a.vendor(),
        vendor_id: a.vendor_id,
        key: a.pci_bus.clone(),
        dedicated_vram: a.dedicated_vram,
        shared_system: a.shared_system,
        integrated: a.integrated,
    }
}

fn engines(e: &Engines) -> EnginesV2 {
    match e {
        Engines::Measured { by_engine } => EnginesV2::Measured {
            busiest: round1(e.busiest().unwrap_or(0.0)),
            by_engine: by_engine
                .iter()
                .filter(|(_, v)| **v != 0.0)
                .map(|(k, v)| (k.clone(), round1(*v)))
                .collect(),
        },
        Engines::Unavailable { reason } => EnginesV2::Unavailable { reason: *reason },
    }
}

fn model_file(m: &MappedModel) -> ModelFileV2 {
    let mut evidence = m.sources.clone();
    evidence.sort();
    ModelFileV2 {
        path: m.path.clone(),
        size: m.size,
        resident: m.resident,
        gguf: m.gguf,
        evidence,
        store_name: m.store_name.clone(),
        kind: m.kind.clone(),
        kind_hint: m.kind_hint.clone(),
    }
}

fn llm(v: &LlmView) -> LlmV2 {
    let models = v
        .models
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let file = v.files.get(i).map(model_file);
            LlmModelV2 {
                path: g.path.clone(),
                name: g.name(),
                label: v.label(i),
                arch: g.arch().to_string(),
                kind: g.kind().to_string(),
                kind_label: v.kind_label(i),
                kind_hint: file
                    .as_ref()
                    .map(|f| f.kind_hint.clone())
                    .unwrap_or_default(),
                decision_type: g.decision_type().map(String::from),
                params: g.n_params,
                quant: g.quant(),
                layers: g.n_layers(),
                ctx_train: g.ctx_train(),
                gpu_weight: Bytes::Estimated {
                    bytes: g.gpu_weight_bytes(None),
                    from: crate::provenance::EstimateSource::Gguf,
                },
                evidence: file
                    .as_ref()
                    .map(|f| f.evidence.clone())
                    .unwrap_or_default(),
                store_name: file.and_then(|f| f.store_name),
            }
        })
        .collect();
    LlmV2 {
        models,
        segments: v
            .segments
            .iter()
            .map(|s| SegmentV2 {
                label: s.label.clone(),
                kind: s.kind,
                bytes: s.bytes,
            })
            .collect(),
        kv_per_token: v.kv_per_token.map(|bytes| Bytes::Estimated {
            bytes,
            from: crate::provenance::EstimateSource::Gguf,
        }),
        n_ctx: v.n_ctx,
        flags: serde_json::to_value(&v.flags).unwrap_or(serde_json::Value::Null),
        server: v.server.clone(),
        log_source: v.log_source.clone(),
        notes: v.notes.clone(),
    }
}

fn resident(row: &ProcRow, backend: Option<BackendKind>) -> Option<ResidentV2> {
    let from = |r: Resident| ResidentV2 {
        vram: r.vram,
        gtt: r.gtt,
        summable: false,
    };
    match row.origin {
        // Their fds cannot be read from this namespace.
        Origin::Outside => Some(from(Resident {
            vram: Bytes::unavailable(Reason::OutsideNamespace),
            gtt: Bytes::unavailable(Reason::OutsideNamespace),
        })),
        Origin::Local if backend == Some(BackendKind::Drm) => row.gpu.resident.map(from),
        Origin::Local | Origin::Windows => None,
    }
}

fn process(row: &ProcRow, backend: Option<BackendKind>) -> ProcessV2 {
    // Committed and paged-out memory are WDDM counters; Linux has neither.
    let not_on_backend = Bytes::unavailable(Reason::NotOnBackend);
    let paged_out = match row.gpu.paged_out() {
        m @ Bytes::Measured { .. } => m,
        _ => not_on_backend,
    };
    ProcessV2 {
        pid: Some(row.pid),
        origin: row.origin,
        name: row.name.clone(),
        kind: row.kind.clone(),
        runtimes: row.detail.runtimes.clone(),
        vram: row.gpu.vram,
        owner_inferred_by: row.gpu.owner,
        shared: row.gpu.shared,
        committed: match row.gpu.committed {
            m @ Bytes::Measured { .. } => m,
            _ => not_on_backend,
        },
        paged_out,
        resident: resident(row, backend),
        engines: engines(&row.gpu.engines),
        ports: row.detail.ports.clone(),
        inside: row.inside(),
        model_files: row.detail.models.iter().map(model_file).collect(),
        installed_models: row.detail.candidates.iter().map(model_file).collect(),
        handy: row.handy.clone(),
        llm: row.llm.as_ref().map(llm),
    }
}

pub fn from_snapshot(s: &Snapshot) -> SnapshotV2 {
    SnapshotV2 {
        schema_version: SCHEMA_VERSION,
        taken: s.taken,
        backend: s.backend,
        adapter: s.adapter.as_ref().map(adapter),
        driver: s.driver.clone(),
        cuda: s.cuda.clone(),
        memory: MemoryV2 {
            total: s.vram_total,
            used: s.vram_used,
            shared_used: s.shared_used,
            attributed: s.attributed(),
            shared_is_spill: s.shared_is_spill,
        },
        status: StatusV2 {
            process: s.process_status.clone(),
            device: s.device_status.map(|r| r.code().to_string()),
            counter: None,
            host: None,
            wsl_split: None,
        },
        device: s.device.clone(),
        notes: s.notes.clone(),
        processes: s.procs.iter().map(|r| process(r, s.backend)).collect(),
    }
}

/// Pretty JSON with a two-space indent.
pub fn to_json(s: &Snapshot) -> String {
    serde_json::to_string_pretty(&from_snapshot(s)).unwrap_or_default()
}
