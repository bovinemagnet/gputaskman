#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::provenance::Bytes;
use gputaskman_tui::action::{Action, Pane, from_key, from_mouse};
use gputaskman_tui::app::{App, CHART_MODES, Popup};
use gputaskman_tui::fixtures::*;
use gputaskman_tui::history::Chart;
use ratatui::crossterm::event::{
    KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::layout::Rect;

fn key(app: &mut App, code: KeyCode) {
    if let Some(a) = from_key(KeyEvent::new(code, KeyModifiers::NONE), app) {
        app.update(a);
    }
}

fn click(column: u16, row: u16) -> MouseEvent {
    MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column,
        row,
        modifiers: KeyModifiers::NONE,
    }
}

#[test]
fn navigation_sort_pause_and_help() {
    let mut app = App::default();
    app.apply_all(vec![snapshot()]);
    assert_eq!(app.rows().len(), 2);
    assert_eq!(app.selected.as_deref(), Some("10")); // the LLM is auto-selected
    key(&mut app, KeyCode::Down);
    assert_eq!(app.selected.as_deref(), Some("20"));
    app.apply_all(vec![snapshot()]);
    assert_eq!(app.selected.as_deref(), Some("20"));
    for _ in 0..5 {
        key(&mut app, KeyCode::Char('s'));
    }
    assert_eq!(app.sort, 0);
    key(&mut app, KeyCode::Char('a'));
    key(&mut app, KeyCode::Char('p'));
    assert!(app.show_all);
    assert!(app.paused);
    key(&mut app, KeyCode::Char('?'));
    assert!(matches!(app.popup, Popup::Help { .. }));
    key(&mut app, KeyCode::Char('p')); // ignored while help is open
    assert!(app.paused);
    key(&mut app, KeyCode::Esc);
    key(&mut app, KeyCode::Char('p'));
    assert_eq!(app.popup, Popup::None);
    assert!(!app.paused);
    key(&mut app, KeyCode::Char('q'));
    assert!(app.quit);
}

#[test]
fn ctrl_c_quits_even_with_a_popup_open() {
    let mut app = App::default();
    app.popup = Popup::Help { scroll: 0 };
    let a = from_key(
        KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
        &app,
    );
    assert_eq!(a, Some(Action::Quit));
}

#[test]
fn down_without_a_selection_starts_from_the_cursor() {
    let mut s = snapshot();
    s.procs[0].llm = None; // nothing to auto-select
    let mut app = App::default();
    app.apply_all(vec![s]);
    assert_eq!(app.selected, None);
    key(&mut app, KeyCode::Down);
    assert_eq!(app.selected.as_deref(), Some("20"));
    key(&mut app, KeyCode::Home);
    assert_eq!(app.selected.as_deref(), Some("10"));
    key(&mut app, KeyCode::End);
    assert_eq!(app.selected.as_deref(), Some("20"));
}

#[test]
fn a_vanished_selection_is_cleared_and_the_cursor_kept() {
    let mut app = App::default();
    app.apply_all(vec![snapshot()]);
    key(&mut app, KeyCode::Down);
    let mut s = snapshot();
    s.procs.truncate(1);
    app.apply_all(vec![s]);
    assert_eq!(app.selected, None);
    assert_eq!(app.cursor, 0); // clamped to the one row left
}

#[test]
fn auto_selection_waits_for_an_llm_and_never_steals_a_choice() {
    let mut no_llm = snapshot();
    no_llm.procs[0].llm = None;
    let mut app = App::default();
    app.apply_all(vec![no_llm.clone()]);
    assert_eq!(app.selected, None);
    app.apply_all(vec![snapshot()]);
    assert_eq!(app.selected.as_deref(), Some("10"));

    let mut chosen = App::default();
    chosen.selected = Some("20".into());
    chosen.apply_all(vec![snapshot()]);
    assert_eq!(chosen.selected.as_deref(), Some("20"));
}

#[test]
fn selection_keys_follow_the_gpu_count() {
    let mut app = App::default();
    app.apply_all(vec![snapshot()]);
    assert_eq!(app.selected.as_deref(), Some("10"));
    app.apply_all(vec![snapshot(), snapshot()]);
    assert_eq!(app.selected.as_deref(), Some("0:10"));
    app.selected = Some("1:10".into());
    app.apply_all(vec![snapshot(), snapshot()]);
    assert_eq!(app.selected.as_deref(), Some("1:10"));
    app.apply_all(vec![snapshot()]);
    assert_eq!(app.selected, None); // "1:" has no single-GPU equivalent
    assert_eq!(app.histories.len(), 2);
}

#[test]
fn chart_cycle_visits_all_then_each_chart() {
    let mut app = App::default();
    assert_eq!(app.chart, None);
    for expected in CHART_MODES.iter().skip(1) {
        key(&mut app, KeyCode::Char('c'));
        assert_eq!(app.chart, *expected);
    }
    key(&mut app, KeyCode::Char('c'));
    assert_eq!(app.chart, None);
}

#[test]
fn chart_menu_is_keyboard_accessible() {
    let mut app = App::default();
    app.update(Action::OpenChartMenu);
    assert_eq!(app.popup, Popup::ChartMenu { index: 0 });
    key(&mut app, KeyCode::Down);
    key(&mut app, KeyCode::Down);
    key(&mut app, KeyCode::Enter);
    assert_eq!(app.chart, Some(Chart::Clock));
    assert_eq!(app.popup, Popup::None);
    app.update(Action::OpenChartMenu);
    assert_eq!(app.popup, Popup::ChartMenu { index: 2 });
    key(&mut app, KeyCode::Home);
    key(&mut app, KeyCode::Enter);
    assert_eq!(app.chart, None);
    app.update(Action::OpenChartMenu);
    key(&mut app, KeyCode::Esc);
    assert_eq!(app.popup, Popup::None);
}

#[test]
fn toggles_and_scrolls_respect_their_limits() {
    let mut app = App::default();
    key(&mut app, KeyCode::Char('m'));
    assert!(app.slim);
    app.areas.dashboard_max_scroll = 4;
    app.update(Action::Scroll(Pane::Dashboard, 3));
    app.update(Action::Scroll(Pane::Dashboard, 3));
    assert_eq!(app.dashboard_scroll, 4);
    app.update(Action::Scroll(Pane::Dashboard, -10));
    assert_eq!(app.dashboard_scroll, 0);
}

#[test]
fn mouse_hits_use_the_areas_from_the_last_draw() {
    let mut app = App::default();
    app.apply_all(vec![snapshot()]);
    app.areas.chart_selector = Rect::new(10, 5, 14, 1);
    app.areas.table_body = Rect::new(0, 20, 100, 10);
    app.areas.footer_hints[1] = Rect::new(8, 40, 6, 1);
    assert_eq!(from_mouse(click(12, 5), &app), Some(Action::OpenChartMenu));
    assert_eq!(from_mouse(click(3, 21), &app), Some(Action::Select(1)));
    assert_eq!(from_mouse(click(9, 40), &app), Some(Action::CycleSort));
    assert_eq!(from_mouse(click(99, 99), &app), None);
    app.update(Action::Select(1));
    assert_eq!(app.selected.as_deref(), Some("20"));
    app.update(Action::Select(9)); // past the end: ignored
    assert_eq!(app.selected.as_deref(), Some("20"));
}

#[test]
fn mouse_in_the_open_chart_menu_picks_or_closes() {
    let mut app = App::default();
    app.update(Action::OpenChartMenu);
    app.areas.chart_menu = Rect::new(10, 6, 18, 11);
    assert_eq!(from_mouse(click(12, 8), &app), Some(Action::MenuPick(1)));
    assert_eq!(from_mouse(click(0, 0), &app), Some(Action::ClosePopup));
    app.update(Action::MenuPick(1));
    assert_eq!(app.chart, Some(Chart::Gpu));
}

#[test]
fn the_wheel_moves_the_table_selection_and_scrolls_other_panes() {
    let mut app = App::default();
    app.apply_all(vec![snapshot()]);
    app.areas.table_body = Rect::new(0, 20, 100, 10);
    app.areas.inspector = Rect::new(0, 31, 100, 5);
    let wheel = |kind, row| MouseEvent {
        kind,
        column: 1,
        row,
        modifiers: KeyModifiers::NONE,
    };
    assert_eq!(
        from_mouse(wheel(MouseEventKind::ScrollDown, 21), &app),
        Some(Action::Scroll(Pane::Table, 1))
    );
    assert_eq!(
        from_mouse(wheel(MouseEventKind::ScrollUp, 32), &app),
        Some(Action::Scroll(Pane::Inspector, -3))
    );
    app.update(Action::Scroll(Pane::Table, 1));
    assert_eq!(app.selected.as_deref(), Some("20"));
}

#[test]
fn sorting_keeps_the_selected_process_selected() {
    let mut s = snapshot();
    s.procs[1].gpu.vram = Bytes::measured(9 * GIB);
    let mut app = App::default();
    app.apply_all(vec![s]);
    app.selected = Some("10".into());
    key(&mut app, KeyCode::Char('s'));
    assert_eq!(app.selected.as_deref(), Some("10"));
}
