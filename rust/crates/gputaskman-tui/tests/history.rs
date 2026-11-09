#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::model::VENDOR_AMD;
use gputaskman_core::provenance::{Bytes, Reason};
use gputaskman_tui::fixtures::*;
use gputaskman_tui::history::*;
use std::collections::BTreeSet;

#[test]
fn chart_palette_uses_distinct_explicit_colours() {
    let colours: BTreeSet<String> = CHARTS.iter().map(|c| format!("{:?}", c.color())).collect();
    assert_eq!(colours.len(), CHARTS.len());
    assert!(
        CHARTS
            .iter()
            .all(|c| matches!(c.color(), ratatui::style::Color::Rgb(..)))
    );
}

#[test]
fn eighth_chart_records_sm_clock_with_scale_and_missing_readings() {
    assert_eq!(CHARTS.len(), 8);
    let mut h = History::default();
    let mut s = snapshot();
    let mut d = device("Test GPU");
    d.sm_clock = Some(1500);
    d.sm_clock_max = Some(3003);
    s.device = Some(d.clone());
    h.record(&s);
    assert_eq!(h.scale(Chart::Clock, Some(&d)), 3003.0);
    assert_eq!(Chart::Clock.unit(), " MHz");
    d.sm_clock = Some(0);
    s.device = Some(d.clone());
    h.record(&s);
    d.sm_clock = None;
    d.sm_clock_max = None;
    s.device = Some(d.clone());
    h.record(&s);
    assert_eq!(h.values(Chart::Clock), &[Some(1500.0), Some(0.0), None]);
    assert_eq!(h.scale(Chart::Clock, Some(&d)), 1500.0);
    s.device = None;
    h.record(&s);
    assert_eq!(
        h.values(Chart::Clock),
        &[Some(1500.0), Some(0.0), None, None]
    );
    assert_eq!(History::default().scale(Chart::Clock, None), 1.0);
}

#[test]
fn unavailable_device_metrics_record_as_missing_and_zero_stays_zero() {
    let mut h = History::default();
    let mut s = snapshot();
    let mut d = device("Test GPU");
    d.util_mem = Some(0.0);
    d.power_w = Some(2.4);
    s.device = Some(d);
    h.record(&s);
    assert_eq!(h.current(Chart::Gpu), None);
    assert_eq!(h.values(Chart::MemActivity), &[Some(0.0)]);
}

#[test]
fn vram_history_needs_both_used_and_total() {
    let mut h = History::default();
    let mut s = snapshot();
    h.record(&s);
    assert_eq!(h.current(Chart::Vram), Some(62.5));
    s.vram_total = Bytes::unavailable(Reason::NotOnBackend);
    h.record(&s);
    assert_eq!(h.current(Chart::Vram), None);
    s.vram_total = Bytes::measured(0);
    h.record(&s);
    assert_eq!(h.current(Chart::Vram), None);
}

#[test]
fn amd_cards_record_the_gfx_clock_and_label_it() {
    let mut h = History::default();
    let mut s = snapshot();
    s.adapter = Some(adapter(VENDOR_AMD, "Radeon"));
    let mut d = device("Radeon");
    d.gfx_clock = Some(2200);
    d.sm_clock = Some(1);
    s.device = Some(d);
    h.record(&s);
    assert!(is_amd(&s));
    assert_eq!(h.current(Chart::Clock), Some(2200.0));
    assert_eq!(Chart::Clock.name(true), "GFX clock");
    assert_eq!(Chart::Clock.name(false), "SM clock");
    assert_eq!(Chart::Clock.menu_name(false, true), "SM/GFX clock");
}

#[test]
fn power_scale_is_the_limit_or_the_highest_reading() {
    let mut h = History::default();
    let mut s = snapshot();
    let mut d = device("GPU");
    d.power_w = Some(200.0);
    d.power_limit_w = Some(150.0);
    s.device = Some(d.clone());
    h.record(&s);
    assert_eq!(h.scale(Chart::Power, Some(&d)), 200.0);
    assert_eq!(h.scale(Chart::Temperature, Some(&d)), 120.0);
    assert_eq!(h.scale(Chart::Gpu, Some(&d)), 100.0);
}

#[test]
fn history_keeps_the_last_sixty_samples() {
    let mut h = History::default();
    let s = snapshot();
    for _ in 0..(CAPACITY + 5) {
        h.record(&s);
    }
    assert_eq!(h.values(Chart::Vram).len(), CAPACITY);
}
