#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::{provenance::*, schema::v1::from_v1};
use serde_json::json;

#[test]
fn v1_rows_map_to_provenance() {
    let v1 = json!({"taken": 1.0, "gpu": "RTX", "backend": "nvml", "vram_total": 100, "vram_used": 50,
        "shared_used": null, "attributed": 30, "shared_is_spill": true, "device": null, "processes": [
        {"pid": 1, "host": null, "name": "a", "kind": "AI", "runtimes": [], "vram_attributed": true,
         "vram_bytes": 30, "vram_inferred_bytes": null, "vram_inferred_by": "order", "shared_bytes": null,
         "committed_bytes": null, "paged_out_bytes": null, "gpu_pct": 5.0, "engines": {"3d": 5.0},
         "ports": [], "inside": "", "model_files": [{"path": "/m", "size": 1, "resident": 0, "gguf": true,
         "sources": ["mapped", "api"], "store_name": null, "kind": "LLM", "kind_hint": ""}], "installed_models": []},
        {"pid": 2, "host": "outside", "name": "x", "kind": "external", "runtimes": [], "vram_attributed": false,
         "vram_bytes": null, "vram_inferred_bytes": 7, "vram_inferred_by": "consistent", "shared_bytes": null,
         "committed_bytes": null, "paged_out_bytes": null, "gpu_pct": null, "engines": null,
         "vram_unattributed_reason": null, "ports": [], "inside": "", "model_files": [], "installed_models": []}]});
    let s = from_v1(&v1).unwrap();
    assert_eq!(s.processes[0].vram, Bytes::measured(30));
    assert_eq!(s.processes[0].owner_inferred_by, Some(Confidence::Order));
    assert_eq!(s.processes[0].model_files[0].evidence.len(), 2);
    assert_eq!(
        s.processes[1].vram,
        Bytes::Inferred {
            bytes: 7,
            by: Confidence::Consistent
        }
    );
    assert_eq!(s.processes[0].shared, Bytes::unavailable(Reason::V1Unknown));
    assert_eq!(s.memory.attributed, Bytes::measured(30));
    assert_eq!(s.memory.shared_used, Bytes::unavailable(Reason::V1Unknown));
}

#[test]
fn missing_keys_are_null_and_drm_reasons_map() {
    let v1 = json!({"processes": [
        {"pid": 3, "name": "d", "vram_bytes": null, "vram_unattributed_reason": "shared_buffers"},
        {"pid": 4, "name": "e"}]});
    let s = from_v1(&v1).unwrap();
    assert_eq!(
        s.processes[0].vram,
        Bytes::unavailable(Reason::SharedBuffers)
    );
    assert_eq!(s.processes[1].vram, Bytes::unavailable(Reason::V1Unknown));
    assert_eq!(s.memory.total, Bytes::unavailable(Reason::V1Unknown));
    assert!(s.adapter.is_none());
    assert!(s.notes.is_empty());
}

#[test]
fn segments_map_exact_to_reported() {
    let v1 = json!({"processes": [{"pid": 1, "name": "l", "llm": {"n_ctx": 4096, "segments": [
        {"label": "w", "bytes": 5, "kind": "weights", "exact": true},
        {"label": "k", "bytes": 6, "kind": "kv", "exact": false}]}}]});
    let s = from_v1(&v1).unwrap();
    let seg = &s.processes[0].llm.as_ref().unwrap().segments;
    assert_eq!(
        seg[0].bytes,
        Bytes::Reported {
            bytes: 5,
            by: Reporter::LoadLog
        }
    );
    assert_eq!(
        seg[1].bytes,
        Bytes::Estimated {
            bytes: 6,
            from: EstimateSource::Gguf
        }
    );
}
