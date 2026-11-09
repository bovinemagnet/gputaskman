//! Text widgets ported from `app.py`: the sparkline and the stacked bar.
use ratatui::style::{Color, Style};
use ratatui::text::Span;

use crate::format::GREY27;

const SPARK: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Python's `spark`: the last `width` values; a missing reading is a blank, zero is the lowest
/// bar, and values are clamped to `0..=top`.
pub fn spark(values: &[Option<f64>], width: usize, top: f64) -> String {
    let top = if top > 0.0 { top } else { 1.0 };
    let start = values.len().saturating_sub(width);
    values[start..]
        .iter()
        .map(|v| match v {
            None => ' ',
            Some(v) => {
                let level = (v / top * 7.0 + 0.5).trunc();
                let level = if level.is_nan() {
                    0.0
                } else {
                    level.clamp(0.0, 7.0)
                };
                SPARK[level as usize]
            }
        })
        .collect()
}

/// Python's `stacked_bar`: a `█` run per part in proportion to `total` (half-to-even rounding,
/// as Python's `round`), never more than `width` in all, then `░` for the rest.
pub fn stacked_bar(parts: &[(f64, Color)], total: f64, width: usize) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut used = 0usize;
    for &(value, color) in parts {
        let n = if total > 0.0 {
            (value / total * width as f64).round_ties_even().max(0.0) as usize
        } else {
            0
        };
        let n = n.min(width - used);
        if n > 0 {
            spans.push(Span::styled("█".repeat(n), Style::new().fg(color)));
            used += n;
        }
    }
    if width > used {
        spans.push(Span::styled(
            "░".repeat(width - used),
            Style::new().fg(GREY27),
        ));
    }
    spans
}
