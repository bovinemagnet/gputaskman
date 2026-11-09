#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::provenance::{Bytes, Reason};
use gputaskman_tui::app::App;
use gputaskman_tui::fixtures::*;
use gputaskman_tui::history::{CHARTS, Chart};
use gputaskman_tui::view::charts::*;

fn grid_rows(app: &App, width: u16) -> usize {
    // Header, grid rows, then one note line per GPU.
    chart_lines(app, width).len() - 1 - app.snapshots.len()
}

#[test]
fn eighth_chart_shows_sm_clock_with_its_scale() {
    let mut s = snapshot();
    let mut d = device("GPU");
    d.sm_clock = Some(1500);
    d.sm_clock_max = Some(3003);
    s.device = Some(d);
    let mut app = App::default();
    app.apply_all(vec![s.clone()]);
    let text = plain(&chart_lines(&app, 117));
    assert!(text.contains("1500 MHz"));
    assert!(text.contains("SM clock 0–3003 MHz"));
    assert_eq!(grid_rows(&app, 117), 4); // 2 columns at 117 wide

    let mut d = s.device.clone().unwrap();
    d.sm_clock = Some(0);
    s.device = Some(d.clone());
    app.apply_all(vec![s.clone()]);
    d.sm_clock = None;
    d.sm_clock_max = None;
    s.device = Some(d);
    app.apply_all(vec![s]);
    app.chart = Some(Chart::Clock);
    let text = plain(&chart_lines(&app, 117));
    assert!(text.contains("N/A"));
    assert!(text.contains("█▁ "), "{text}");
}

#[test]
fn chart_cycle_shows_the_selected_metric_alone_and_preserves_gaps() {
    let mut s = snapshot();
    let mut d = device("GPU");
    d.util_gpu = Some(50.0);
    d.util_mem = Some(0.0);
    d.temp = Some(49.0);
    d.power_w = Some(72.0);
    d.power_limit_w = Some(150.0);
    d.dec = Some(0.0);
    s.device = Some(d);
    let mut app = App::default();
    app.apply_all(vec![s]);
    let all = plain(&chart_lines(&app, 180));
    for c in CHARTS {
        assert!(all.contains(c.name(false)), "{} missing", c.name(false));
    }
    for c in CHARTS {
        app.chart = Some(c);
        let one = chart_lines(&app, 180);
        assert_eq!(grid_rows(&app, 180), 1);
        assert!(plain(&one).contains(c.name(false)));
        if c == Chart::Encoder {
            assert!(plain(&one).contains("N/A"));
        }
        if c == Chart::Decoder {
            assert!(plain(&one).contains("0%"));
            assert!(plain(&one).contains('▁'));
        }
    }
}

#[test]
fn chart_layout_reflows_with_the_width() {
    for (width, columns) in [(180, 3), (117, 2), (80, 1), (60, 1), (40, 1)] {
        assert_eq!(grid_columns(8, width), columns, "width {width}");
    }
    assert_eq!(grid_columns(1, 180), 1);
    let mut s = snapshot();
    let mut d = device("GPU");
    d.power_w = Some(200.0);
    d.power_limit_w = Some(150.0);
    s.device = Some(d);
    let mut app = App::default();
    app.apply_all(vec![s]);
    for (width, rows) in [(180, 3), (117, 4), (80, 8)] {
        assert_eq!(grid_rows(&app, width), rows, "width {width}");
    }
    assert!(plain(&chart_lines(&app, 180)).contains("0–200 W"));
}

#[test]
fn all_charts_work_without_device_telemetry_or_vram_capacity() {
    let mut s = snapshot();
    s.vram_total = Bytes::unavailable(Reason::NotOnBackend);
    let mut app = App::default();
    app.apply_all(vec![s]);
    let lines = chart_lines(&app, 117);
    let grid = plain(&lines[1..lines.len() - 1]);
    assert_eq!(grid.matches("N/A").count(), 8);
    assert!(!grid.contains('▁'));
}

#[test]
fn several_gpus_get_their_own_grids_and_notes() {
    let mut app = App::default();
    app.apply_all(vec![snapshot(), snapshot()]);
    let text = plain(&chart_lines(&app, 140));
    assert!(text.contains("GPU 0"));
    assert!(text.contains("GPU 1"));
    assert!(text.contains("GPU 0: Recent samples"));
    assert!(text.contains("GPU 1: Recent samples"));
}

#[test]
fn the_selector_names_the_mode_and_the_menu_lists_every_mode() {
    let mut app = App::default();
    app.apply_all(vec![snapshot()]);
    assert_eq!(selector_label(&app), "All charts");
    let header = plain(&chart_lines(&app, 140)[..1]);
    assert_eq!(header, "Charts  [All charts ▾]  Click to choose · c cycles");
    assert_eq!(selector_width(&app), 14);
    app.chart = Some(Chart::Clock);
    assert_eq!(selector_label(&app), "SM clock");
    let entries = menu_entries(&app);
    assert_eq!(entries.len(), 9);
    assert_eq!(entries[0], "All charts");
    assert_eq!(entries[2], "SM clock");
    app.apply_all(vec![snapshot(), snapshot()]);
    assert_eq!(menu_entries(&app)[2], "SM/GFX clock");
}

#[test]
fn before_the_first_sample_only_the_header_is_drawn() {
    let app = App::default();
    assert_eq!(chart_lines(&app, 120).len(), 1);
}
