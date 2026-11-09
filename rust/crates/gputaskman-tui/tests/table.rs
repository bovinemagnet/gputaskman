#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::collections::BTreeMap;

use gputaskman_core::model::{BackendKind, Engines, ProcGpu};
use gputaskman_core::provenance::{Bytes, Confidence, Reason};
use gputaskman_tui::app::App;
use gputaskman_tui::fixtures::*;
use gputaskman_tui::format::{GREY62, RED, YELLOW};
use gputaskman_tui::view::table::{cell, cells, subtitle};
use ratatui::style::Modifier;

fn app_with(s: gputaskman_core::collector::Snapshot) -> App {
    let mut app = App::default();
    app.apply_all(vec![s]);
    app
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
    let app = app_with(s);
    assert_eq!(cell(&app, "10", "GPU%").as_deref(), Some("65"));
    assert_eq!(cell(&app, "10", "3D").as_deref(), Some("65"));
    assert_eq!(cell(&app, "10", "VRAM").as_deref(), Some("n/a"));
    assert_eq!(cell(&app, "10", "Spill").as_deref(), Some("")); // NVML has no spill counter
    assert_eq!(cell(&app, "10", "Paged").as_deref(), Some(""));
}

#[test]
fn unknown_engines_print_na_in_every_engine_column() {
    let mut s = snapshot();
    s.procs[0].gpu.engines = Engines::Unavailable {
        reason: Reason::SharedClient,
    };
    let app = app_with(s);
    for column in ["GPU%", "3D", "Copy", "Video"] {
        assert_eq!(cell(&app, "10", column).as_deref(), Some("n/a"), "{column}");
    }
}

#[test]
fn drm_spill_follows_provenance_and_turns_red_for_big_llm_spill() {
    let mut s = snapshot();
    s.backend = Some(BackendKind::Drm);
    s.procs[0].gpu.shared = Bytes::measured(300 * MIB);
    let mut unknown = row(30, "shared", "app", ProcGpu::measured(0));
    unknown.gpu.vram = Bytes::unavailable(Reason::SharedClient);
    unknown.gpu.shared = Bytes::unavailable(Reason::SharedClient);
    s.procs.push(unknown);
    let app = app_with(s);
    assert_eq!(cell(&app, "10", "Spill").as_deref(), Some("300 MB"));
    assert_eq!(cell(&app, "30", "Spill").as_deref(), Some("n/a"));
    assert_eq!(cell(&app, "30", "Paged").as_deref(), Some("")); // ruling R4
    let rows = app.rows();
    let llm = rows.iter().find(|r| r.key() == "10").unwrap();
    let spill = &cells(llm, false)[5];
    assert_eq!(spill.1.fg, Some(RED));
    assert!(spill.1.add_modifier.contains(Modifier::BOLD));
}

#[test]
fn small_spill_is_yellow_and_on_an_apu_never_red() {
    let mut s = snapshot();
    s.backend = Some(BackendKind::Drm);
    s.shared_is_spill = false;
    s.procs[0].gpu.shared = Bytes::measured(300 * MIB);
    let app = app_with(s);
    let rows = app.rows();
    assert_eq!(cells(&rows[0], false)[5].1.fg, Some(YELLOW));
}

#[test]
fn inferred_owners_keep_their_marker_and_grey_unknown_vram() {
    let mut s = snapshot();
    s.procs[0].gpu.owner = Some(Confidence::Order);
    let mut unknown = row(30, "unknown", "app", ProcGpu::measured(0));
    unknown.gpu.vram = Bytes::unavailable(Reason::Incomplete);
    s.procs.push(unknown);
    let app = app_with(s);
    assert_eq!(cell(&app, "10", "VRAM").as_deref(), Some("8.00 GB?"));
    let rows = app.rows();
    let unknown = rows.iter().find(|r| r.key() == "30").unwrap();
    assert_eq!(cells(unknown, false)[4].1.fg, Some(GREY62));
}

#[test]
fn columns_show_runtime_kind_and_inside() {
    let mut s = snapshot();
    s.procs[0].detail.runtimes = vec![
        "llama.cpp".into(),
        "ggml".into(),
        "CUDA".into(),
        "cuBLAS".into(),
    ];
    let app = app_with(s);
    assert_eq!(
        cell(&app, "10", "Runtime").as_deref(),
        Some("llama.cpp ggml CUDA")
    );
    assert_eq!(cell(&app, "10", "Kind").as_deref(), Some("LLM"));
    assert_eq!(
        cell(&app, "20", "What's inside").as_deref(),
        Some("desktop back buffers + composed frames")
    );
    assert_eq!(cell(&app, "10", "PID").as_deref(), Some("10"));
}

#[test]
fn several_gpus_add_a_gpu_column() {
    let first = snapshot();
    let mut second = snapshot();
    second.procs[0].gpu.vram = Bytes::measured(3 * GIB);
    second.procs[0].gpu.engines = Engines::Unavailable {
        reason: Reason::NvmlNotLoaded,
    };
    let mut app = App::default();
    app.apply_all(vec![first, second]);
    assert_eq!(app.rows().len(), 4);
    assert_eq!(cell(&app, "0:10", "GPU").as_deref(), Some("0"));
    assert_eq!(cell(&app, "1:10", "GPU").as_deref(), Some("1"));
    assert_eq!(cell(&app, "1:10", "VRAM").as_deref(), Some("3.00 GB"));
    assert_eq!(cell(&app, "1:10", "GPU%").as_deref(), Some("n/a"));
    app.apply_all(vec![snapshot()]);
    assert_eq!(cell(&app, "10", "GPU"), None);
}

#[test]
fn explicit_gpu_keeps_single_device_layout() {
    let app = app_with(snapshot());
    assert!(!app.multi());
    assert_eq!(cell(&app, "10", "GPU"), None);
    assert_eq!(cells(&app.rows()[0], false).len(), 12);
}

#[test]
fn the_subtitle_names_the_sort_and_filter() {
    let mut app = App::default();
    assert_eq!(subtitle(&app), "sort: VRAM  ·  top processes");
    app.sort = 1;
    app.show_all = true;
    assert_eq!(subtitle(&app), "sort: GPU%  ·  all processes");
}

#[test]
fn likely_kind_is_explicit_in_the_kind_column() {
    let mut s = snapshot();
    s.procs[0].kind = "decision?".into();
    let app = app_with(s);
    assert_eq!(cell(&app, "10", "Kind").as_deref(), Some("decision?"));
}
