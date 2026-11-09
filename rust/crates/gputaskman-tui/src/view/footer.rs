//! The key-hint footer; each hint is clickable.
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::action::FOOTER;
use crate::format::{ACCENT, bold};

pub fn draw(f: &mut Frame, area: Rect) -> [Rect; 7] {
    let mut rects = [Rect::default(); 7];
    let mut spans = Vec::new();
    let mut x = area.x;
    for (i, (key, label, _)) in FOOTER.iter().enumerate() {
        let width = u16::try_from(key.chars().count() + 1 + label.chars().count()).unwrap_or(0);
        spans.push(Span::styled(*key, bold().fg(ACCENT)));
        spans.push(Span::raw(format!(" {label}  ")));
        rects[i] = Rect::new(x, area.y, width, 1).intersection(area);
        x = x.saturating_add(width + 2);
    }
    f.render_widget(
        Paragraph::new(Line::from(spans)).style(Style::new().bg(Color::Indexed(236))),
        area,
    );
    rects
}
