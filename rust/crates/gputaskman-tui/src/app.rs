//! Everything the screen shows besides the snapshots themselves: selection, sort, toggles,
//! popups, scroll positions and where things were last drawn.
use gputaskman_core::collector::Snapshot;
use ratatui::layout::Rect;

use crate::history::{CHARTS, Chart, History};
use crate::rows::{self, ShownRow};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Popup {
    #[default]
    None,
    Help {
        scroll: u16,
    },
    /// The chart selector's list; `index` into `CHART_MODES`.
    ChartMenu {
        index: usize,
    },
}

/// Where things were drawn last time: mouse hits, paging and scroll limits use these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Areas {
    pub dashboard: Rect,
    pub dashboard_max_scroll: u16,
    pub chart_selector: Rect,
    pub chart_menu: Rect,
    /// The table's data rows (below its header).
    pub table_body: Rect,
    /// Index of the first row drawn in `table_body`.
    pub table_offset: usize,
    pub inspector: Rect,
    pub inspector_max_scroll: u16,
    pub help_max_scroll: u16,
    pub footer_hints: [Rect; 7],
}

/// The chart selector's entries: all charts, then each chart.
pub const CHART_MODES: [Option<Chart>; 9] = [
    None,
    Some(CHARTS[0]),
    Some(CHARTS[1]),
    Some(CHARTS[2]),
    Some(CHARTS[3]),
    Some(CHARTS[4]),
    Some(CHARTS[5]),
    Some(CHARTS[6]),
    Some(CHARTS[7]),
];

pub(crate) fn mode_index(chart: Option<Chart>) -> usize {
    CHART_MODES.iter().position(|m| *m == chart).unwrap_or(0)
}

#[derive(Debug, Default)]
pub struct App {
    pub snapshots: Vec<Snapshot>,
    /// One per GPU index ever seen.
    pub histories: Vec<History>,
    /// The selected row's `ShownRow::key()`.
    pub selected: Option<String>,
    /// The table position navigation starts from; follows the selection while there is one.
    pub cursor: usize,
    pub sort: usize,
    pub show_all: bool,
    pub paused: bool,
    pub slim: bool,
    /// `None` shows all charts.
    pub chart: Option<Chart>,
    pub popup: Popup,
    pub dashboard_scroll: u16,
    pub inspector_scroll: u16,
    pub areas: Areas,
    pub quit: bool,
    /// Set once an LLM has been auto-selected or the user has chosen a row.
    auto_selected: bool,
}

impl App {
    pub fn multi(&self) -> bool {
        self.snapshots.len() > 1
    }

    /// The table's rows, filtered and sorted as shown.
    pub fn rows(&self) -> Vec<ShownRow<'_>> {
        rows::table_rows(&self.snapshots, self.show_all, self.sort)
    }

    pub fn selected_row(&self) -> Option<ShownRow<'_>> {
        let key = self.selected.as_deref()?;
        self.rows().into_iter().find(|r| r.key() == key)
    }

    /// Takes one refresh: a snapshot per GPU, in GPU order.
    pub fn apply_all(&mut self, snaps: Vec<Snapshot>) {
        if snaps.is_empty() {
            return;
        }
        let multi = snaps.len() > 1;
        if !self.snapshots.is_empty() && multi != self.multi() {
            self.selected = self.selected.take().and_then(|k| {
                if multi {
                    Some(format!("0:{k}"))
                } else {
                    k.strip_prefix("0:").map(String::from)
                }
            });
        }
        if self.histories.len() < snaps.len() {
            self.histories.resize_with(snaps.len(), History::default);
        }
        for (history, snap) in self.histories.iter_mut().zip(&snaps) {
            history.record(snap);
        }
        self.snapshots = snaps;
        // Ruling R2: the first LLM is selected once, and never over a user's choice.
        if self.selected.is_none()
            && !self.auto_selected
            && let Some(key) = rows::first_llm_key(&self.snapshots)
        {
            self.selected = Some(key);
        }
        if self.selected.is_some() {
            self.auto_selected = true;
        }
        self.sync_selection();
    }

    /// Keeps `cursor` on the selected row, or clears a selection whose row is gone (ruling R3).
    pub(crate) fn sync_selection(&mut self) {
        let (found, n) = {
            let rows = self.rows();
            let found = self
                .selected
                .as_deref()
                .map(|k| rows.iter().position(|r| r.key() == k));
            (found, rows.len())
        };
        match found {
            Some(Some(i)) => self.cursor = i,
            Some(None) => self.selected = None,
            None => {}
        }
        self.cursor = self.cursor.min(n.saturating_sub(1));
    }
}
