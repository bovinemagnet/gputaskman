//! Drawing: one module per part of the screen, laid out by `draw`.
pub mod charts;
pub mod footer;
pub mod header;
pub mod help;
pub mod inspector;
pub mod table;

use ratatui::Frame;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::Line;
use ratatui::widgets::{
    Block, BorderType, Clear, List, ListItem, ListState, Padding, Paragraph, Widget, Wrap,
};

use crate::app::{App, Areas, Popup};
use crate::format::{ACCENT, GREY50, fg};

pub const MIN_WIDTH: u16 = 40;
pub const MIN_HEIGHT: u16 = 10;

/// `value * numerator / denominator` without u16 overflow on very large screens.
pub fn scaled(value: u16, numerator: u32, denominator: u32) -> u16 {
    u16::try_from(u32::from(value) * numerator / denominator).unwrap_or(u16::MAX)
}

/// A rounded, padded box around wrapped lines, and the rows it needs at `width`.
fn boxed(
    lines: Vec<Line<'static>>,
    title: String,
    width: u16,
    border: ratatui::style::Color,
) -> (Paragraph<'static>, u16) {
    let text = Paragraph::new(lines).wrap(Wrap { trim: false });
    let height = u16::try_from(text.line_count(width.saturating_sub(4).max(1)))
        .unwrap_or(u16::MAX)
        .saturating_add(2);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(fg(border))
        .padding(Padding::horizontal(1))
        .title(title);
    (text.block(block), height)
}

/// Paints content `content_height` rows tall off-screen and copies rows `scroll..` into `area`.
fn render_scrolled(
    f: &mut Frame,
    area: Rect,
    content_height: u16,
    scroll: u16,
    paint: impl FnOnce(Rect, &mut Buffer),
) {
    let full = Rect::new(0, 0, area.width, content_height.max(area.height));
    let mut tmp = Buffer::empty(full);
    paint(full, &mut tmp);
    let buf = f.buffer_mut();
    for y in 0..area.height {
        for x in 0..area.width {
            buf[(area.x + x, area.y + y)] = tmp[(x, y + scroll)].clone();
        }
    }
}

pub fn draw(app: &App, f: &mut Frame) -> Areas {
    let area = f.area();
    let mut areas = Areas::default();
    if area.width < MIN_WIDTH || area.height < MIN_HEIGHT {
        f.render_widget(Paragraph::new("terminal too small"), area);
        return areas;
    }
    let width = area.width;

    let (device, device_h) = boxed(
        header::device_lines(app, width),
        header::device_title(app).to_string(),
        width,
        ACCENT,
    );
    let (charts, charts_h) = boxed(
        charts::chart_lines(app, width),
        String::new(),
        width,
        GREY50,
    );
    let dash_content = device_h.saturating_add(charts_h);
    let dash_h = dash_content.min(area.height / 2);

    let (inspector, inspector_h) = boxed(
        inspector::lines(app, width),
        inspector::title(app),
        width,
        GREY50,
    );
    let inspector_h = inspector_h.min(scaled(area.height, 3, 10)).max(3);

    let [dash, table_area, insp, footer_area] = Layout::vertical([
        Constraint::Length(dash_h),
        Constraint::Min(4),
        Constraint::Length(inspector_h),
        Constraint::Length(1),
    ])
    .areas(area);

    // Dashboard: a scroll view over the two boxes.
    areas.dashboard = dash;
    areas.dashboard_max_scroll = dash_content.saturating_sub(dash.height);
    let scroll = app.dashboard_scroll.min(areas.dashboard_max_scroll);
    render_scrolled(f, dash, dash_content, scroll, |full, buf| {
        device.render(
            Rect {
                height: device_h,
                ..full
            },
            buf,
        );
        charts.render(
            Rect {
                y: device_h,
                height: charts_h,
                ..full
            },
            buf,
        );
    });
    // The selector sits on the charts box's first content row, after its label.
    let selector_row = i32::from(device_h) + 1 - i32::from(scroll);
    if (0..i32::from(dash.height)).contains(&selector_row) {
        let label = u16::try_from(charts::LABEL.chars().count()).unwrap_or(0);
        areas.chart_selector = Rect::new(
            dash.x + 2 + label,
            dash.y + u16::try_from(selector_row).unwrap_or(0),
            charts::selector_width(app),
            1,
        )
        .intersection(dash);
    }

    let (body, offset) = table::draw(app, f, table_area);
    areas.table_body = body;
    areas.table_offset = offset;

    areas.inspector = insp;
    let inner_rows = insp.height.saturating_sub(2);
    let content_rows = u16::try_from(
        Paragraph::new(inspector::lines(app, width))
            .wrap(Wrap { trim: false })
            .line_count(width.saturating_sub(4).max(1)),
    )
    .unwrap_or(u16::MAX);
    areas.inspector_max_scroll = content_rows.saturating_sub(inner_rows);
    f.render_widget(
        inspector.scroll((app.inspector_scroll.min(areas.inspector_max_scroll), 0)),
        insp,
    );

    areas.footer_hints = footer::draw(f, footer_area);

    match app.popup {
        Popup::None => {}
        Popup::Help { scroll } => areas.help_max_scroll = help::draw(app, f, area, scroll),
        Popup::ChartMenu { index } => {
            let entries = charts::menu_entries(app);
            let w = u16::try_from(
                entries
                    .iter()
                    .map(|e| e.chars().count())
                    .max()
                    .unwrap_or(10)
                    + 4,
            )
            .unwrap_or(20);
            let h = u16::try_from(entries.len() + 2).unwrap_or(11);
            let anchor = if areas.chart_selector.width > 0 {
                areas.chart_selector
            } else {
                dash
            };
            let menu = Rect::new(anchor.x, anchor.y.saturating_add(1), w, h).intersection(area);
            let items: Vec<ListItem<'static>> = entries.into_iter().map(ListItem::new).collect();
            let list = List::new(items)
                .block(Block::bordered().border_style(fg(ACCENT)))
                .highlight_style(
                    ratatui::style::Style::new().bg(ratatui::style::Color::Indexed(25)),
                );
            let mut state = ListState::default().with_selected(Some(index));
            f.render_widget(Clear, menu);
            f.render_stateful_widget(list, menu, &mut state);
            areas.chart_menu = menu;
        }
    }
    areas
}
