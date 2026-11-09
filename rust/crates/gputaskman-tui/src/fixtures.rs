//! Builders for this crate's tests (feature `test-support`): a port of the Python tests'
//! `snapshot()` and friends.
#![allow(clippy::expect_used, clippy::unwrap_used)]
use std::collections::BTreeMap;

use gputaskman_core::collector::{Origin, ProcRow, Snapshot};
use gputaskman_core::gguf::{GgufInfo, MetaValue};
use gputaskman_core::llm::flags::Flags;
use gputaskman_core::llm::servers::ServerInfo;
use gputaskman_core::llm::view::LlmView;
use gputaskman_core::model::{
    Adapter, BackendKind, DeviceTelemetry, ProcDetail, ProcGpu, VENDOR_AMD,
};
use gputaskman_core::provenance::Bytes;

pub const GIB: u64 = 1 << 30;
pub const MIB: u64 = 1 << 20;

/// A local process row with measured-style defaults from `gpu`.
pub fn row(pid: u32, name: &str, kind: &str, gpu: ProcGpu) -> ProcRow {
    ProcRow {
        pid,
        name: name.into(),
        kind: kind.into(),
        gpu,
        detail: ProcDetail::new(pid, name),
        llm: None,
        handy: None,
        origin: Origin::Local,
    }
}

/// An LLM view with only a llama-server probe result, like the Python fixture's.
pub fn server_view() -> LlmView {
    LlmView {
        models: Vec::new(),
        files: Vec::new(),
        other_files: Vec::new(),
        flags: Flags::default(),
        server: Some(ServerInfo {
            kind: "llama-server".into(),
            url: "http://127.0.0.1:8080".into(),
            n_ctx: None,
            slots_total: None,
            slots_busy: None,
            gen_tok_s: None,
            prompt_tok_s: None,
            model_path: None,
            models: Vec::new(),
        }),
        host: None,
        log_source: None,
        segments: Vec::new(),
        notes: Vec::new(),
        kv_per_token: None,
        kv_type: String::new(),
        n_ctx: None,
    }
}

/// Port of the Python tests' `snapshot()`: NVML, 10 of 16 GiB used, 1 GiB shared, an 8 GiB LLM
/// server (PID 10) and a 2 GiB compositor (PID 20).
pub fn snapshot() -> Snapshot {
    let mut llm = row(10, "llama-server", "LLM", ProcGpu::measured(8 * GIB));
    llm.detail.cmdline = vec![
        "server".into(),
        "--api-key".into(),
        "private-test-value".into(),
    ];
    llm.llm = Some(server_view());
    let compositor = row(20, "Xorg", "compositor", ProcGpu::measured(2 * GIB));
    Snapshot {
        taken: 1.0,
        adapter: None,
        adapters: Vec::new(),
        device: None,
        device_status: None,
        driver: None,
        cuda: None,
        vram_total: Bytes::measured(16 * GIB),
        vram_used: Bytes::measured(10 * GIB),
        shared_used: Bytes::measured(GIB),
        procs: vec![llm, compositor],
        backend: Some(BackendKind::Nvml),
        process_status: None,
        shared_is_spill: true,
        notes: Vec::new(),
    }
}

/// NVML device telemetry with every reading unknown.
pub fn device(name: &str) -> DeviceTelemetry {
    DeviceTelemetry {
        name: name.into(),
        source: "NVML".into(),
        ..DeviceTelemetry::default()
    }
}

pub fn adapter(vendor_id: u16, name: &str) -> Adapter {
    Adapter {
        key: 1,
        pci_bus: "0000:01:00.0".into(),
        name: name.into(),
        vendor_id,
        dedicated_vram: 16 * GIB,
        shared_system: 0,
        integrated: false,
        backend: if vendor_id == VENDOR_AMD {
            BackendKind::Drm
        } else {
            BackendKind::Nvml
        },
    }
}

/// A two-layer causal model header named `name`.
pub fn gguf(arch: &str, name: &str) -> GgufInfo {
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        MetaValue::Str(arch.into()),
    );
    metadata.insert("general.name".to_string(), MetaValue::Str(name.into()));
    metadata.insert(format!("{arch}.block_count"), MetaValue::U(2));
    metadata.insert(format!("{arch}.context_length"), MetaValue::U(4096));
    GgufInfo {
        path: format!("/models/{name}.gguf"),
        file_size: GIB,
        version: 3,
        metadata,
        layer_bytes: BTreeMap::new(),
        output_bytes: 0,
        token_embd_bytes: 0,
        tensor_bytes: GIB,
        n_params: Some(752_000_000),
        error: None,
    }
}
