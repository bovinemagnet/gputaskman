#![allow(clippy::unwrap_used, clippy::expect_used)]
use chrono::{Local, TimeZone, Utc};
use gputaskman_core::env::Paths;
use gputaskman_core::handy::{HandyEvidence, HandyProbe};
use std::fs;

const SELECTED: &str = "handy-computer/nemotron/model-Q8_0.gguf";

fn setup(log: &str) -> (tempfile::TempDir, Paths) {
    let dir = tempfile::tempdir().unwrap();
    let paths = Paths::rooted(dir.path());
    let base = paths.xdg_data.join("com.pais.handy");
    fs::create_dir_all(base.join("logs")).unwrap();
    fs::write(
        base.join("settings_store.json"),
        format!(r#"{{"settings": {{"selected_model": "{SELECTED}"}}}}"#),
    )
    .unwrap();
    fs::write(base.join("logs/handy.log"), log).unwrap();
    (dir, paths)
}

fn started() -> f64 {
    Utc.with_ymd_and_hms(2026, 11, 2, 23, 0, 0)
        .unwrap()
        .timestamp() as f64
}

#[test]
fn selected_model_is_distinct_from_runtime_load_and_unload_events() {
    let log = "[2026-11-02][22:59:00][transcription][INFO] Loaded whisper model 'old.gguf'\n\
[2026-11-02][23:01:00][transcription][INFO] Loaded whisper model 'handy-computer/nemotron/model-Q8_0.gguf' (backend 'Vulkan0')\n\
[2026-11-02][23:01:01][transcription][INFO] Live streaming transcription started (model 'handy-computer/nemotron/model-Q8_0.gguf', backend 'Vulkan0')\n\
[2026-11-02][23:01:02][transcription][INFO] Model idle for 3606s (limit: 3600s), unloading\n\
[2026-11-02][23:01:03][transcription][INFO] Model unloaded due to inactivity (took 20ms)\n\
[2026-11-02][23:01:04][transcription][INFO] Partial transcription: private dictation text\n";
    let (_dir, paths) = setup(log);
    let mut probe = HandyProbe::new(paths);
    let ev = probe.inspect(started()).unwrap();
    assert_eq!(ev.selected_model.as_deref(), Some(SELECTED));
    assert_eq!(ev.last_model.as_deref(), Some(SELECTED));
    assert_eq!(ev.last_event.as_deref(), Some("unloaded"));
    assert_eq!(
        ev.last_event_at.as_deref(),
        Some("2026-11-02T23:01:03+00:00")
    );
    let local = Utc
        .with_ymd_and_hms(2026, 11, 2, 23, 1, 3)
        .unwrap()
        .with_timezone(&Local)
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();
    assert_eq!(ev.last_event_local(), local);
    assert!(!format!("{ev:?}").contains("private dictation text"));
}

#[test]
fn selected_model_without_fresh_log_is_not_reported_as_loaded() {
    let log = "[2026-11-02][22:59:00][transcription][INFO] Loaded whisper model 'old.gguf'\n";
    let (_dir, paths) = setup(log);
    let mut probe = HandyProbe::new(paths);
    let ev = probe.inspect(started()).unwrap();
    assert_eq!(ev.selected_model.as_deref(), Some(SELECTED));
    assert_eq!(ev.last_model, None);
    assert_eq!(ev.last_event, None);
}

#[test]
fn local_formatting_edge_cases() {
    let none = HandyEvidence::default();
    assert_eq!(none.last_event_local(), "");
    let bad = HandyEvidence {
        last_event_at: Some("garbage".into()),
        ..Default::default()
    };
    assert_eq!(bad.last_event_local(), "garbage");
}

#[test]
fn nothing_found_returns_none() {
    let dir = tempfile::tempdir().unwrap();
    let mut probe = HandyProbe::new(Paths::rooted(dir.path()));
    assert_eq!(probe.inspect(0.0), None);
}
