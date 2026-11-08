#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::model::*;
use gputaskman_core::provenance::*;

fn adapter(key: u64, vendor_id: u16, vram: u64) -> Adapter {
    Adapter {
        key,
        pci_bus: String::new(),
        name: format!("gpu{key}"),
        vendor_id,
        dedicated_vram: vram,
        shared_system: 0,
        integrated: false,
        backend: BackendKind::Drm,
    }
}

#[test]
fn pci_key_matches_python() {
    assert_eq!(pci_key("00000000:01:00.0"), 0x1000);
    assert_eq!(pci_key("0000:03:00.0"), 0x3000);
    assert_eq!(pci_key(""), 0);
}

#[test]
fn nvidia_first_then_vram_then_key() {
    let ordered = ordered_adapters(vec![
        adapter(3, VENDOR_AMD, 16 * GIB),
        adapter(1, VENDOR_NVIDIA, 8 * GIB),
        adapter(2, VENDOR_AMD, 32 * GIB),
    ]);
    assert_eq!(
        ordered.iter().map(|a| a.key).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
}

#[test]
fn adapter_indices_do_not_clamp_to_another_gpu() {
    let list = [adapter(1, VENDOR_NVIDIA, GIB)];
    assert_eq!(select_adapter(&list, 0).unwrap().unwrap().key, 1);
    let err = select_adapter(&list, 1).unwrap_err();
    assert_eq!(
        err.to_string(),
        "--gpu 1 is out of range; 1 GPU adapter(s) available"
    );
    assert!(select_adapter(&[], 0).unwrap().is_none());
    assert!(select_adapter(&[], 1).is_err());
}

#[test]
fn best_vram_keeps_measured_and_inferred_zero_distinct_from_unknown() {
    // Port of test_core.py MemoryTests.test_best_vram_keeps_measured_and_inferred_zero_distinct_from_unknown
    let mut g = ProcGpu::measured(0);
    assert_eq!(g.shown_vram().map(Shown::get), Some(0));
    g.vram = Bytes::Inferred {
        bytes: 8 * GIB,
        by: Confidence::Consistent,
    };
    assert_eq!(g.shown_vram().map(Shown::get), Some(8 * GIB));
    g.vram = Bytes::Inferred {
        bytes: 0,
        by: Confidence::Consistent,
    };
    assert_eq!(g.shown_vram().map(Shown::get), Some(0));
    g.vram = Bytes::unavailable(Reason::NvmlHidden);
    assert_eq!(g.shown_vram(), None);
}

#[test]
fn paged_out_is_clamped_commit_difference_and_unknown_without_commit() {
    let mut g = ProcGpu::measured(4 * GIB);
    assert!(matches!(g.paged_out(), Bytes::Unavailable { .. }));
    g.shared = Bytes::measured(GIB);
    g.committed = Bytes::measured(6 * GIB);
    assert_eq!(g.paged_out(), Bytes::measured(GIB));
    g.committed = Bytes::measured(GIB);
    assert_eq!(g.paged_out(), Bytes::measured(0));
}

#[test]
fn live_thresholds_match_python() {
    assert!(!ProcGpu::measured(MIB - 1).is_live());
    assert!(ProcGpu::measured(MIB).is_live());
    let mut hidden = ProcGpu::measured(0);
    hidden.vram = Bytes::unavailable(Reason::NvmlHidden);
    assert!(hidden.is_live());
    let mut busy = ProcGpu::measured(0);
    busy.engines = Engines::Measured {
        by_engine: [("3d".to_string(), 0.5)].into(),
    };
    assert!(busy.is_live());
    let mut spill = ProcGpu::measured(0);
    spill.shared = Bytes::measured(16 * MIB);
    assert!(spill.is_live());
}

#[test]
fn attribution_total_rules() {
    let hidden = ProcGpu {
        vram: Bytes::unavailable(Reason::NvmlHidden),
        ..ProcGpu::measured(0)
    };
    let rows = [ProcGpu::measured(3 * GIB), hidden];
    assert_eq!(attributed_total(&rows), Bytes::measured(3 * GIB)); // hidden NVML counts as 0, like Python
    let drm = ProcGpu {
        vram: Bytes::unavailable(Reason::SharedBuffers),
        ..ProcGpu::measured(0)
    };
    let rows = [ProcGpu::measured(3 * GIB), drm];
    assert_eq!(
        attributed_total(&rows),
        Bytes::unavailable(Reason::SharedBuffers)
    );
    let inferred = ProcGpu {
        vram: Bytes::Inferred {
            bytes: GIB,
            by: Confidence::Order,
        },
        ..ProcGpu::measured(0)
    };
    let rows = [ProcGpu::measured(GIB), inferred];
    assert_eq!(attributed_total(&rows), Bytes::measured(GIB)); // inferred bytes are never summed
}

#[test]
fn evidence_ladder_and_labels() {
    assert!(
        Evidence::Api < Evidence::Mapped
            && Evidence::Mapped < Evidence::Cmdline
            && Evidence::Cmdline < Evidence::Installed
    );
    let mut m = MappedModel {
        path: "/m.gguf".into(),
        size: 1,
        resident: 0,
        gguf: true,
        sources: vec![Evidence::Mapped],
        store_name: None,
        kind: "LLM".into(),
        kind_hint: String::new(),
    };
    assert_eq!(m.evidence_label(), "in use (mapped)");
    m.add_source(Evidence::Api);
    m.add_source(Evidence::Api);
    assert_eq!(m.sources, vec![Evidence::Mapped, Evidence::Api]);
    assert_eq!(m.evidence_label(), "confirmed (api+mapped)");
    m.sources = vec![Evidence::Installed];
    assert_eq!(m.evidence_label(), "installed, not proven loaded");
    m.kind_hint = "decision model".into();
    assert_eq!(m.kind_label(), "likely decision model (name)");
}
