#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::collections::BTreeMap;

use gputaskman_core::collector::Origin;
use gputaskman_core::handy::HandyEvidence;
use gputaskman_core::llm::view::{Segment, SegmentKind};
use gputaskman_core::model::{BackendKind, Engines, Evidence, MappedModel, ProcGpu};
use gputaskman_core::provenance::{Bytes, Confidence, EstimateSource, Reason, Reporter, Resident};
use gputaskman_core::report::{DRM_RESIDENT_NOTE, ORDER_NOTE};
use gputaskman_tui::app::App;
use gputaskman_tui::fixtures::*;
use gputaskman_tui::view::inspector::{PROMPT, lines, title};

fn selected(s: gputaskman_core::collector::Snapshot, key: &str) -> App {
    let mut app = App::default();
    app.selected = Some(key.to_string());
    app.apply_all(vec![s]);
    app
}

fn text(app: &App) -> String {
    plain(&lines(app, 180))
}

#[test]
fn without_a_selection_the_inspector_prompts() {
    let mut s = snapshot();
    s.procs[0].llm = None;
    let mut app = App::default();
    app.apply_all(vec![s]);
    assert_eq!(text(&app), PROMPT);
    assert_eq!(title(&app), "");
}

#[test]
fn native_hidden_vram_keeps_the_engine_reading() {
    let mut s = snapshot();
    let mut g = ProcGpu::measured(0);
    g.vram = Bytes::unavailable(Reason::NvmlHidden);
    g.engines = Engines::Measured {
        by_engine: BTreeMap::from([("compute".to_string(), 65.0)]),
    };
    s.procs = vec![row(10, "python3", "AI", g)];
    let app = selected(s, "10");
    let first = plain(&lines(&app, 180)[..1]);
    assert!(first.contains("compute 65%"), "{first}");
    assert!(first.contains("spill(RAM) n/a")); // ruling R5
    assert_eq!(title(&app), "python3 (10) - AI");
}

#[test]
fn handy_evidence_is_explicit() {
    let mut s = snapshot();
    let mut handy = row(10, "handy", "AI", ProcGpu::measured(GIB));
    handy.handy = Some(HandyEvidence {
        selected_model: Some("handy-computer/nemotron/model.gguf".into()),
        last_model: Some("model.gguf".into()),
        last_event: Some("loaded".into()),
        last_event_at: Some("2026-11-02T23:01:00+00:00".into()),
    });
    s.procs[0] = handy;
    let app = selected(s, "10");
    let t = text(&app);
    assert!(t.contains("Handy selected (settings; not load proof)"));
    assert!(t.contains("runtime-reported history, not current GPU residency"));
}

#[test]
fn likely_kind_and_evidence_are_explicit_in_the_model_table() {
    let mut s = snapshot();
    let mut view = server_view();
    view.models = vec![gguf("qwen35", "Snapshot_Cfg")];
    view.files = vec![MappedModel {
        path: "/models/OpenJev.gguf".into(),
        size: 1000,
        resident: 0,
        gguf: true,
        sources: vec![Evidence::Mapped],
        store_name: None,
        kind: String::new(),
        kind_hint: "decision model".into(),
    }];
    s.procs[0].llm = Some(view);
    s.procs[0].kind = "decision?".into();
    let app = selected(s, "10");
    let t = text(&app);
    assert!(t.contains("likely decision model (name)"), "{t}");
    assert!(t.contains("in use (mapped)"));
    assert!(t.contains("4,096"));
    assert!(t.contains("server llama-server http://127.0.0.1:8080"));
}

#[test]
fn llm_segments_mark_exact_sizes_and_the_log_source() {
    let mut s = snapshot();
    let mut view = server_view();
    view.segments = vec![
        Segment {
            label: "weights (CUDA0)".into(),
            bytes: Bytes::Reported {
                bytes: 604 * MIB,
                by: Reporter::LoadLog,
            },
            kind: SegmentKind::Weights,
        },
        Segment {
            label: "KV cache".into(),
            bytes: Bytes::Estimated {
                bytes: 448 * MIB,
                from: EstimateSource::Gguf,
            },
            kind: SegmentKind::Kv,
        },
    ];
    view.log_source = Some("/var/log/llama/server.log".into());
    view.notes = vec!["! low context".into(), "KV cost 112 KB/token".into()];
    s.procs[0].llm = Some(view);
    let app = selected(s, "10");
    let ls = lines(&app, 180);
    let t = plain(&ls);
    assert!(t.contains("weights (CUDA0) 604 MB"));
    assert!(t.contains("KV cache 448 MB"));
    assert!(t.contains("(allocated sizes from server.log)"));
    assert!(t.contains("• ! low context"));
    let exact = style_of(&ls, "weights (CUDA0)").unwrap();
    assert!(exact.add_modifier.contains(ratatui::style::Modifier::BOLD));
    let estimate = style_of(&ls, "KV cache 448 MB").unwrap();
    assert!(
        !estimate
            .add_modifier
            .contains(ratatui::style::Modifier::BOLD)
    );
}

#[test]
fn drm_rows_explain_unknown_vram_resident_bytes_and_kfd() {
    let mut s = snapshot();
    s.backend = Some(BackendKind::Drm);
    let mut g = ProcGpu::measured(0);
    g.vram = Bytes::unavailable(Reason::SharedClient);
    g.resident = Some(Resident {
        vram: Bytes::measured(GIB),
        gtt: Bytes::unavailable(Reason::Legacy),
    });
    g.kfd = true;
    s.procs = vec![row(30, "rocm-app", "AI", g)];
    let app = selected(s, "30");
    let t = text(&app);
    assert!(t.contains("VRAM and GPU% n/a: this process holds a DRM client"));
    assert!(t.contains(&format!(
        "Resident VRAM 1.00 GB, GTT n/a. {DRM_RESIDENT_NOTE}"
    )));
    assert!(t.contains("Engines n/a: this process uses ROCm/HIP (/dev/kfd)"));
}

#[test]
fn inferred_owners_get_the_matching_note_and_host_rows_say_so() {
    let mut s = snapshot();
    s.procs[0].gpu.owner = Some(Confidence::Order);
    let mut outside = row(
        99,
        "(outside this container)",
        "external",
        ProcGpu::measured(GIB),
    );
    outside.origin = Origin::Outside;
    s.procs.push(outside);
    let app = selected(s.clone(), "10");
    assert!(text(&app).contains(ORDER_NOTE));
    let app = selected(s, "o99");
    assert_eq!(
        title(&app),
        "(outside this container) (host PID 99) - external"
    );
}

#[test]
fn other_rows_explain_their_kind_files_candidates_ports_and_cmdline() {
    let mut s = snapshot();
    let comp = &mut s.procs[1];
    comp.detail.ports = vec![8091, 8092];
    comp.detail.candidates = vec![MappedModel {
        path: "/models/qwen.gguf".into(),
        size: GIB,
        resident: 0,
        gguf: true,
        sources: vec![Evidence::Installed],
        store_name: Some("qwen3:4b".into()),
        kind: "LLM".into(),
        kind_hint: String::new(),
    }];
    comp.detail.access_denied = true;
    let app = selected(s, "20");
    let t = text(&app);
    assert!(t.contains("Installed candidates (not proven loaded):"));
    assert!(t.contains("  qwen3:4b  (1.00 GB, LLM)"));
    assert!(t.contains("listening: 8091, 8092"));
    assert!(t.contains("(protected process - run elevated for runtime/file details)"));
}

#[test]
fn the_llm_command_line_is_shown_truncated() {
    let app = selected(snapshot(), "10");
    assert!(text(&app).contains("server --api-key private-test-value"));
}

#[test]
fn several_gpus_name_the_gpu_and_warn_about_placement() {
    let mut app = App::default();
    app.apply_all(vec![snapshot(), snapshot()]);
    app.selected = Some("1:10".into());
    assert_eq!(title(&app), "llama-server (10) - LLM - GPU 1");
    assert!(text(&app).contains("not a measured per-device model placement"));
}
