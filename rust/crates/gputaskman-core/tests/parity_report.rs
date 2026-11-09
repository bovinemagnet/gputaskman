#![cfg(feature = "parity")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Report parity with the Python implementation: a scenario snapshot goes through Python's
//! `report.to_dict`, then the Rust v1 reader, and must equal the Rust v2 snapshot built from the same
//! rows, ignoring `taken`. Needs `uv` and the synced Python environment.
use gputaskman_core::collector::{Origin, ProcRow, Snapshot};
use gputaskman_core::model::{BackendKind, Engines, ProcDetail, ProcGpu};
use gputaskman_core::provenance::{Bytes, Confidence, Reason, Resident};
use gputaskman_core::schema::v1::from_v1;
use gputaskman_core::schema::v2::{EnginesV2, SnapshotV2, from_snapshot};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap()
}

fn python(scenario: &Path) -> Value {
    let out = Command::new("uv")
        .args(["run", "--locked", "python", "rust/parity/run_python.py"])
        .arg(scenario)
        .current_dir(repo_root())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

/// The amdgpu codes Python writes in `vram_unattributed_reason`.
fn drm_reason(code: &str) -> Reason {
    match code {
        "legacy" => Reason::Legacy,
        "shared_client" => Reason::SharedClient,
        "shared_buffers" => Reason::SharedBuffers,
        "primary" => Reason::Primary,
        "incomplete" => Reason::Incomplete,
        "partial_scan" => Reason::PartialScan,
        other => panic!("unknown drm reason {other}"),
    }
}

fn bytes_or(v: &Value, missing: Reason) -> Bytes {
    v.as_u64()
        .map_or(Bytes::unavailable(missing), Bytes::measured)
}

/// The Rust row for one scenario row: what the NVML or DRM backend would have produced for it.
fn rust_gpu(r: &Value, drm: bool) -> ProcGpu {
    let not_on_backend = Bytes::unavailable(Reason::NotOnBackend);
    let reason = r["drm_reason"].as_str().map(drm_reason);
    let attributed = r["attributed"].as_bool().unwrap_or(true) && reason.is_none();
    let dedicated = r["dedicated"].as_u64().unwrap_or(0);
    let vram = match (attributed, reason) {
        (true, _) => Bytes::measured(dedicated),
        (false, Some(reason)) => Bytes::unavailable(reason),
        (false, None) => Bytes::unavailable(Reason::NvmlHidden),
    };
    let shared = match (drm, reason) {
        (false, _) => not_on_backend,
        (true, Some(reason)) => Bytes::unavailable(reason),
        (true, None) => Bytes::measured(r["shared"].as_u64().unwrap_or(0)),
    };
    let by_engine = r["engines"]
        .as_object()
        .map(|o| {
            o.iter()
                .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
                .collect()
        })
        .unwrap_or_default();
    let missing = if reason == Some(Reason::Incomplete) {
        Reason::Incomplete
    } else {
        Reason::Legacy
    };
    ProcGpu {
        vram,
        shared,
        committed: not_on_backend,
        engines: Engines::Measured { by_engine },
        owner: match r["owner"].as_str() {
            Some("consistent") => Some(Confidence::Consistent),
            Some("order") => Some(Confidence::Order),
            _ => None,
        },
        resident: drm.then(|| Resident {
            vram: bytes_or(&r["resident_vram"], missing),
            gtt: bytes_or(&r["resident_gtt"], missing),
        }),
        kfd: false,
    }
}

fn rust_snapshot(s: &Value) -> Snapshot {
    let drm = s["backend"] == "drm";
    let unknown = Bytes::unavailable(Reason::NotOnBackend);
    let procs = s["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            let pid = u32::try_from(r["pid"].as_u64().unwrap()).unwrap();
            let name = r["name"].as_str().unwrap();
            ProcRow {
                pid,
                name: name.to_string(),
                kind: r["kind"].as_str().unwrap().to_string(),
                gpu: rust_gpu(r, drm),
                detail: ProcDetail::new(pid, name),
                llm: None,
                handy: None,
                origin: Origin::Local,
            }
        })
        .collect();
    Snapshot {
        taken: 1.0,
        adapter: None,
        adapters: vec![],
        device: None,
        device_status: None,
        driver: None,
        cuda: None,
        vram_total: Bytes::measured(s["vram_total"].as_u64().unwrap_or(0)),
        vram_used: s["vram_used"].as_u64().map_or(unknown, Bytes::measured),
        shared_used: s["shared_used"].as_u64().map_or(unknown, Bytes::measured),
        procs,
        backend: Some(if drm {
            BackendKind::Drm
        } else {
            BackendKind::Nvml
        }),
        process_status: None,
        shared_is_spill: true,
        notes: vec![],
    }
}

/// Project a v2 snapshot onto what a v1 document can say. v1 writes `null` for every unknown value
/// and names a reason only for amdgpu VRAM (`vram_unattributed_reason`), so the v1 reader turns the
/// other unknowns into `v1_unknown`. Here the Rust side loses the same detail: unknown stays unknown,
/// only the reason v1 cannot carry is dropped. Resident counters are not projected, because the v1
/// reader rebuilds them by the same rule `from_snapshot` uses; the scenario gives every DRM row
/// explicit resident values, since v1 cannot say why a null one is missing.
fn as_v1_can_say(mut s: SnapshotV2) -> SnapshotV2 {
    let v1_unknown = |b: &mut Bytes| {
        if b.reason().is_some() {
            *b = Bytes::unavailable(Reason::V1Unknown);
        }
    };
    let m = &mut s.memory;
    for b in [
        &mut m.total,
        &mut m.used,
        &mut m.shared_used,
        &mut m.attributed,
    ] {
        v1_unknown(b);
    }
    for p in &mut s.processes {
        let drm_vram_reason = matches!(
            p.vram.reason(),
            Some(
                Reason::Legacy
                    | Reason::SharedClient
                    | Reason::SharedBuffers
                    | Reason::Primary
                    | Reason::Incomplete
                    | Reason::PartialScan
            )
        );
        if !(drm_vram_reason && s.backend == Some(BackendKind::Drm)) {
            v1_unknown(&mut p.vram);
        }
        for b in [&mut p.shared, &mut p.committed, &mut p.paged_out] {
            v1_unknown(b);
        }
        if matches!(p.engines, EnginesV2::Unavailable { .. }) {
            p.engines = EnginesV2::Unavailable {
                reason: Reason::V1Unknown,
            };
        }
    }
    s
}

#[test]
fn report_matches_python() {
    let path = repo_root().join("rust/parity/scenarios/report_mixed.json");
    let scenario: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let mut expected = from_v1(&python(&path)).unwrap();
    let mut actual = as_v1_can_say(from_snapshot(&rust_snapshot(&scenario)));
    expected.taken = Some(0.0);
    actual.taken = Some(0.0);
    assert_eq!(actual, expected);
}
