//! Input: keys and mouse events become `Action`s, and `App::update` applies them.
use ratatui::crossterm::event::{
    KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::layout::{Position, Rect};

use crate::app::{App, CHART_MODES, Popup, mode_index};
use crate::rows::SORTS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pane {
    Dashboard,
    Table,
    Inspector,
    Help,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Quit,
    CycleSort,
    CycleChart,
    ToggleAll,
    ToggleSlim,
    TogglePause,
    Help,
    ClosePopup,
    Up,
    Down,
    PageUp,
    PageDown,
    Home,
    End,
    /// Select the table row at this index.
    Select(usize),
    OpenChartMenu,
    MenuMove(i32),
    MenuChoose,
    /// Choose `CHART_MODES[i]` directly (a click).
    MenuPick(usize),
    Scroll(Pane, i32),
}

/// Footer hints: key, label, action.
pub const FOOTER: [(&str, &str, Action); 7] = [
    ("q", "Quit", Action::Quit),
    ("s", "Sort", Action::CycleSort),
    ("c", "Chart", Action::CycleChart),
    ("a", "All/Top", Action::ToggleAll),
    ("m", "Slim/Full", Action::ToggleSlim),
    ("p", "Pause", Action::TogglePause),
    ("?", "Explain", Action::Help),
];

const MENU_LAST: i32 = CHART_MODES.len() as i32;

pub fn from_key(key: KeyEvent, app: &App) -> Option<Action> {
    if key.kind != KeyEventKind::Press {
        return None;
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return Some(Action::Quit);
    }
    // Letter bindings are for plain keys; Ctrl or Alt chords belong to the terminal.
    let held = key
        .modifiers
        .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT);
    match app.popup {
        Popup::Help { .. } => match key.code {
            KeyCode::Esc => Some(Action::ClosePopup),
            KeyCode::Char('q' | '?') if !held => Some(Action::ClosePopup),
            KeyCode::Up => Some(Action::Scroll(Pane::Help, -1)),
            KeyCode::Down => Some(Action::Scroll(Pane::Help, 1)),
            KeyCode::PageUp => Some(Action::Scroll(Pane::Help, -10)),
            KeyCode::PageDown => Some(Action::Scroll(Pane::Help, 10)),
            _ => None,
        },
        Popup::ChartMenu { .. } => match key.code {
            KeyCode::Up => Some(Action::MenuMove(-1)),
            KeyCode::Down => Some(Action::MenuMove(1)),
            KeyCode::Home => Some(Action::MenuMove(-MENU_LAST)),
            KeyCode::End => Some(Action::MenuMove(MENU_LAST)),
            KeyCode::Enter => Some(Action::MenuChoose),
            KeyCode::Esc => Some(Action::ClosePopup),
            _ => None,
        },
        Popup::None if held => None,
        Popup::None => match key.code {
            KeyCode::Char('q') => Some(Action::Quit),
            KeyCode::Char('s') => Some(Action::CycleSort),
            KeyCode::Char('c') => Some(Action::CycleChart),
            KeyCode::Char('a') => Some(Action::ToggleAll),
            KeyCode::Char('m') => Some(Action::ToggleSlim),
            KeyCode::Char('p') => Some(Action::TogglePause),
            KeyCode::Char('?') => Some(Action::Help),
            KeyCode::Up => Some(Action::Up),
            KeyCode::Down => Some(Action::Down),
            KeyCode::PageUp => Some(Action::PageUp),
            KeyCode::PageDown => Some(Action::PageDown),
            KeyCode::Home => Some(Action::Home),
            KeyCode::End => Some(Action::End),
            _ => None,
        },
    }
}

pub fn from_mouse(m: MouseEvent, app: &App) -> Option<Action> {
    let at = |r: Rect| r.contains(Position::new(m.column, m.row));
    let a = &app.areas;
    match m.kind {
        MouseEventKind::ScrollDown | MouseEventKind::ScrollUp => {
            let sign = if m.kind == MouseEventKind::ScrollDown {
                1
            } else {
                -1
            };
            if matches!(app.popup, Popup::Help { .. }) {
                Some(Action::Scroll(Pane::Help, 3 * sign))
            } else if matches!(app.popup, Popup::ChartMenu { .. }) {
                None
            } else if at(a.table_body) {
                Some(Action::Scroll(Pane::Table, sign)) // ruling R1
            } else if at(a.inspector) {
                Some(Action::Scroll(Pane::Inspector, 3 * sign))
            } else if at(a.dashboard) {
                Some(Action::Scroll(Pane::Dashboard, 3 * sign))
            } else {
                None
            }
        }
        MouseEventKind::Down(MouseButton::Left) => match app.popup {
            Popup::Help { .. } => None,
            Popup::ChartMenu { .. } => {
                if at(a.chart_menu) {
                    let i = usize::from(m.row - a.chart_menu.y).checked_sub(1)?;
                    (i < CHART_MODES.len()).then_some(Action::MenuPick(i))
                } else {
                    Some(Action::ClosePopup)
                }
            }
            Popup::None => {
                if at(a.chart_selector) {
                    Some(Action::OpenChartMenu)
                } else if at(a.table_body) {
                    Some(Action::Select(
                        a.table_offset + usize::from(m.row - a.table_body.y),
                    ))
                } else {
                    FOOTER
                        .iter()
                        .zip(a.footer_hints)
                        .find(|(_, r)| at(*r))
                        .map(|((_, _, action), _)| *action)
                }
            }
        },
        _ => None,
    }
}

fn step(value: u16, delta: i32, max: u16) -> u16 {
    let v = i64::from(value) + i64::from(delta);
    u16::try_from(v.clamp(0, i64::from(max))).unwrap_or(max)
}

impl App {
    pub fn update(&mut self, action: Action) {
        let page = i32::from(self.areas.table_body.height.max(1));
        match action {
            Action::Quit => self.quit = true,
            Action::CycleSort => {
                self.sort = (self.sort + 1) % SORTS.len();
                self.sync_selection();
            }
            Action::CycleChart => {
                self.chart = CHART_MODES[(mode_index(self.chart) + 1) % CHART_MODES.len()];
            }
            Action::ToggleAll => {
                self.show_all = !self.show_all;
                self.sync_selection();
            }
            Action::ToggleSlim => self.slim = !self.slim,
            Action::TogglePause => self.paused = !self.paused,
            Action::Help => self.popup = Popup::Help { scroll: 0 },
            Action::ClosePopup => self.popup = Popup::None,
            Action::Up => self.move_by(-1),
            Action::Down => self.move_by(1),
            Action::PageUp => self.move_by(-page),
            Action::PageDown => self.move_by(page),
            Action::Home => self.move_to(0),
            Action::End => self.move_to(usize::MAX),
            Action::Select(i) => {
                if i < self.rows().len() {
                    self.move_to(i);
                }
            }
            Action::OpenChartMenu => {
                self.popup = Popup::ChartMenu {
                    index: mode_index(self.chart),
                };
            }
            Action::MenuMove(delta) => {
                if let Popup::ChartMenu { index } = &mut self.popup {
                    let last = CHART_MODES.len() as i64 - 1;
                    let next = (*index as i64 + i64::from(delta)).clamp(0, last);
                    *index = usize::try_from(next).unwrap_or(0);
                }
            }
            Action::MenuChoose => {
                if let Popup::ChartMenu { index } = self.popup {
                    self.chart = CHART_MODES[index.min(CHART_MODES.len() - 1)];
                    self.popup = Popup::None;
                }
            }
            Action::MenuPick(i) => {
                if let Some(mode) = CHART_MODES.get(i) {
                    self.chart = *mode;
                    self.popup = Popup::None;
                }
            }
            Action::Scroll(Pane::Table, delta) => self.move_by(delta),
            Action::Scroll(Pane::Dashboard, delta) => {
                self.dashboard_scroll = step(
                    self.dashboard_scroll,
                    delta,
                    self.areas.dashboard_max_scroll,
                );
            }
            Action::Scroll(Pane::Inspector, delta) => {
                self.inspector_scroll = step(
                    self.inspector_scroll,
                    delta,
                    self.areas.inspector_max_scroll,
                );
            }
            Action::Scroll(Pane::Help, delta) => {
                if let Popup::Help { scroll } = &mut self.popup {
                    *scroll = step(*scroll, delta, self.areas.help_max_scroll);
                }
            }
        }
    }

    fn move_by(&mut self, delta: i32) {
        let n = self.rows().len();
        if n == 0 {
            return;
        }
        let base = self
            .selected
            .as_deref()
            .and_then(|k| self.rows().iter().position(|r| r.key() == k))
            .unwrap_or(self.cursor.min(n - 1));
        let next = (base as i64 + i64::from(delta)).clamp(0, n as i64 - 1);
        self.move_to(usize::try_from(next).unwrap_or(0));
    }

    fn move_to(&mut self, index: usize) {
        let key = {
            let rows = self.rows();
            if rows.is_empty() {
                return;
            }
            let i = index.min(rows.len() - 1);
            (i, rows[i].key())
        };
        if self.selected.as_deref() != Some(key.1.as_str()) {
            self.inspector_scroll = 0;
        }
        self.cursor = key.0;
        self.selected = Some(key.1);
    }
}
