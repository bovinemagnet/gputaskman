#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_tui::action::{Action, from_key, from_mouse};
use gputaskman_tui::app::{App, Popup};
use gputaskman_tui::fixtures::*;
use gputaskman_tui::history::Chart;
use ratatui::crossterm::event::{
    KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};

fn click_at(app: &mut App, (column, row): (u16, u16)) {
    let ev = MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column,
        row,
        modifiers: KeyModifiers::NONE,
    };
    if let Some(a) = from_mouse(ev, app) {
        app.update(a);
    }
}

fn press(app: &mut App, code: KeyCode) {
    if let Some(a) = from_key(KeyEvent::new(code, KeyModifiers::NONE), app) {
        app.update(a);
    }
}

fn with_device() -> gputaskman_core::collector::Snapshot {
    let mut s = snapshot();
    let mut d = device("Test GPU");
    d.util_gpu = Some(50.0);
    d.temp = Some(49.0);
    d.power_w = Some(200.0);
    d.power_limit_w = Some(150.0);
    s.device = Some(d);
    s
}

#[test]
fn the_screen_shows_every_part() {
    let mut app = App::default();
    app.apply_all(vec![with_device()]);
    let t = render(&mut app, 180, 50);
    let text = screen(&t);
    for part in [
        "Full diagnostics",
        "VRAM",
        "Charts",
        "[All charts ▾]",
        "PID",
        "What's inside",
        "llama-server",
        "sort: VRAM  ·  top processes",
        "llama-server (10) - LLM",
        "q Quit",
        "? Explain",
    ] {
        assert!(text.contains(part), "{part} missing from\n{text}");
    }
}

#[test]
fn chart_layout_reflows_while_paused_and_keeps_processes_visible() {
    let mut app = App::default();
    app.apply_all(vec![with_device()]);
    press(&mut app, KeyCode::Down);
    press(&mut app, KeyCode::Char('p'));
    for (width, height) in [
        (180, 50),
        (117, 59),
        (80, 40),
        (60, 24),
        (80, 24),
        (180, 50),
    ] {
        let t = render(&mut app, width, height);
        let a = app.areas;
        assert!(
            a.table_body.height + 2 >= 4,
            "{width}x{height}: table too short"
        );
        assert!(
            a.inspector.bottom() < height,
            "{width}x{height}: inspector over footer"
        );
        assert_eq!(app.selected.as_deref(), Some("20"));
        assert!(app.paused);
        assert!(screen(&t).contains("[PAUSED]") || a.dashboard_max_scroll > 0);
        if height == 24 {
            assert!(
                a.dashboard_max_scroll > 0,
                "{width}x{height}: dashboard should scroll"
            );
            app.update(Action::Scroll(gputaskman_tui::action::Pane::Dashboard, 100));
            let t = render(&mut app, width, height);
            assert!(screen(&t).contains("Decoder"), "{width}x{height}");
            app.update(Action::Scroll(
                gputaskman_tui::action::Pane::Dashboard,
                -100,
            ));
        }
    }
}

#[test]
fn chart_dropdown_is_clickable_and_keyboard_accessible() {
    let mut app = App::default();
    app.apply_all(vec![snapshot()]);
    let t = render(&mut app, 117, 59);
    let at = find(&t, "[All charts ▾]").unwrap();
    click_at(&mut app, (at.0 + 2, at.1));
    assert_eq!(app.popup, Popup::ChartMenu { index: 0 });
    let t = render(&mut app, 117, 59);
    assert!(screen(&t).contains("Mem activity"));
    assert!(app.areas.chart_menu.height >= 11);
    press(&mut app, KeyCode::Down);
    press(&mut app, KeyCode::Down);
    press(&mut app, KeyCode::Enter);
    assert_eq!(app.chart, Some(Chart::Clock));
    let t = render(&mut app, 117, 59);
    assert!(screen(&t).contains("[SM clock ▾]"));
    let at = find(&t, "[SM clock ▾]").unwrap();
    click_at(&mut app, at);
    render(&mut app, 117, 59);
    let menu = app.areas.chart_menu;
    click_at(&mut app, (menu.x + 2, menu.y + 1)); // "All charts"
    assert_eq!(app.chart, None);
    assert!(screen(&render(&mut app, 117, 59)).contains("[All charts ▾]"));
}

#[test]
fn clicking_a_row_selects_it_and_footer_hints_work() {
    let mut app = App::default();
    app.apply_all(vec![snapshot()]);
    let t = render(&mut app, 140, 42);
    let at = find(&t, "Xorg").unwrap();
    click_at(&mut app, at);
    assert_eq!(app.selected.as_deref(), Some("20"));
    let t = render(&mut app, 140, 42);
    assert!(screen(&t).contains("Xorg (20) - compositor"));
    let sort = find(&t, "s Sort").unwrap();
    click_at(&mut app, sort);
    assert_eq!(app.sort, 1);
    let help = find(&t, "? Explain").unwrap();
    click_at(&mut app, help);
    assert!(matches!(app.popup, Popup::Help { .. }));
}

#[test]
fn help_opens_scrolls_and_closes() {
    let mut app = App::default();
    app.apply_all(vec![snapshot()]);
    press(&mut app, KeyCode::Char('?'));
    let t = render(&mut app, 140, 42);
    let text = screen(&t);
    assert!(text.contains("What can be known about GPU memory?"));
    assert!(text.contains("Per-process"));
    assert!(app.areas.help_max_scroll > 0);
    press(&mut app, KeyCode::PageDown);
    assert!(matches!(app.popup, Popup::Help { scroll } if scroll > 0));
    press(&mut app, KeyCode::Char('q'));
    assert_eq!(app.popup, Popup::None);
    assert!(!app.quit);
}

#[test]
fn a_tiny_terminal_says_so() {
    let mut app = App::default();
    app.apply_all(vec![snapshot()]);
    let t = render(&mut app, 39, 20);
    assert!(screen(&t).contains("terminal too small"));
    let t = render(&mut app, 80, 9);
    assert!(screen(&t).contains("terminal too small"));
}

#[test]
fn page_keys_move_by_the_visible_rows() {
    let mut s = snapshot();
    for pid in 100..160 {
        s.procs.push(row(
            pid,
            &format!("p{pid}"),
            "app",
            gputaskman_core::model::ProcGpu::measured(64 * MIB),
        ));
    }
    let mut app = App::default();
    app.apply_all(vec![s]);
    render(&mut app, 140, 42);
    let page = usize::from(app.areas.table_body.height);
    press(&mut app, KeyCode::PageDown);
    assert_eq!(app.cursor, page);
    // Selecting a short process shrinks the content-sized inspector, so the table grows:
    // page again by the new visible height to scroll.
    render(&mut app, 140, 42);
    let page = usize::from(app.areas.table_body.height);
    press(&mut app, KeyCode::PageDown);
    assert_eq!(app.cursor, 19 + page);
    let t = render(&mut app, 140, 42);
    assert!(app.areas.table_offset > 0);
    let key = app.selected.clone().unwrap();
    let name = app
        .rows()
        .iter()
        .find(|r| r.key() == key)
        .unwrap()
        .row
        .name
        .clone();
    assert!(screen(&t).contains(&name));
}

#[test]
fn several_gpus_show_both_devices_and_a_gpu_column() {
    let mut app = App::default();
    app.apply_all(vec![snapshot(), snapshot()]);
    let t = render(&mut app, 140, 42);
    let text = screen(&t);
    assert!(text.contains("GPU 0"));
    assert!(text.contains("GPU 1"));
    app.apply_all(vec![snapshot()]);
    render(&mut app, 140, 42);
    assert_eq!(app.rows().len(), 2);
}

#[test]
fn before_the_first_sample_the_screen_still_draws() {
    let mut app = App::default();
    let t = render(&mut app, 120, 40);
    assert!(screen(&t).contains("Sampling GPU counters..."));
}

#[test]
fn the_help_popup_survives_a_very_wide_screen() {
    let mut app = App::default();
    app.apply_all(vec![snapshot()]);
    press(&mut app, KeyCode::Char('?'));
    let t = render(&mut app, 700, 40);
    assert!(screen(&t).contains("What can be known about GPU memory?"));
    let top = screen(&t).lines().nth(3).unwrap().to_string();
    let left = top.chars().position(|c| c == '┏').unwrap();
    let right = top.chars().position(|c| c == '┓').unwrap();
    assert!(right - left < 140, "popup is {} wide", right - left + 1);
}

#[test]
fn a_very_tall_screen_draws() {
    let mut app = App::default();
    app.apply_all(vec![with_device()]);
    let t = render(&mut app, 240, 800);
    assert!(screen(&t).contains("q Quit"));
    press(&mut app, KeyCode::Char('?'));
    render(&mut app, 240, 800);
}

#[test]
fn the_spec_size_screen_shows_every_part_in_three_chart_columns() {
    let mut app = App::default();
    app.apply_all(vec![with_device()]);
    let t = render(&mut app, 240, 50);
    let text = screen(&t);
    for part in [
        "Full diagnostics",
        "Charts",
        "PID",
        "What's inside",
        "q Quit",
        "? Explain",
    ] {
        assert!(text.contains(part), "{part} missing from\n{text}");
    }
    assert_eq!(gputaskman_tui::view::charts::grid_columns(8, 240), 3);
    let row = text.lines().find(|l| l.contains("SM clock")).unwrap();
    assert!(row.contains("GPU ") && row.contains("VRAM"), "{row}");
    assert!(!row.contains("Mem activity"), "{row}");
    assert!(app.areas.table_body.height >= 4);
}

#[test]
fn the_chart_menu_is_shifted_to_fit_and_its_rows_stay_clickable() {
    let (w, h) = (80, 14);
    let mut app = App::default();
    app.apply_all(vec![with_device()]);
    render(&mut app, w, h);
    let max = app.areas.dashboard_max_scroll;
    assert!(max > 0);
    let mut near_bottom = false;
    for scroll in 0..=max {
        app.popup = Popup::None;
        app.dashboard_scroll = scroll;
        let t = render(&mut app, w, h);
        let Some(at) = find(&t, "[All charts ▾]") else {
            continue;
        };
        near_bottom |= at.1 + 1 >= app.areas.dashboard.bottom() - 1;
        click_at(&mut app, at);
        let t = render(&mut app, w, h);
        let menu = app.areas.chart_menu;
        assert_eq!(menu.height, 11, "scroll {scroll}: {menu:?}");
        assert!(menu.bottom() <= h && menu.right() <= w, "{menu:?}");
        let text = screen(&t);
        for entry in ["All charts", "Decoder", "Mem activity"] {
            assert!(text.contains(entry), "{entry} missing from\n{text}");
        }
        click_at(&mut app, (menu.x + 2, menu.y + 1));
        assert_eq!(app.chart, None, "scroll {scroll}");
        assert_eq!(app.popup, Popup::None);
    }
    assert!(
        near_bottom,
        "the selector never reached the dashboard's last row"
    );
}
