//! Colours and the provenance-aware text shared by every view. Colour names are Rich's, mapped to
//! xterm-256 indices; the chart colours (in `history`) are exact RGB.
use gputaskman_core::llm::gb;
use gputaskman_core::llm::view::SegmentKind;
use gputaskman_core::provenance::{Bytes, Reason};
use ratatui::style::{Color, Modifier, Style};

pub const MIB: u64 = 1 << 20;

pub const GREY27: Color = Color::Indexed(238);
pub const GREY35: Color = Color::Indexed(240);
pub const GREY42: Color = Color::Indexed(242);
pub const GREY50: Color = Color::Indexed(244);
pub const GREY58: Color = Color::Indexed(246);
pub const GREY62: Color = Color::Indexed(247);
pub const GREY70: Color = Color::Indexed(249);
pub const MAGENTA: Color = Color::Indexed(5);
pub const YELLOW: Color = Color::Indexed(3);
pub const CYAN: Color = Color::Indexed(6);
pub const GREEN: Color = Color::Indexed(2);
pub const RED: Color = Color::Indexed(1);
pub const DARK_ORANGE: Color = Color::Indexed(208);
pub const WHITE: Color = Color::Indexed(7);
/// Border of the device box and key hints (Textual's accent).
pub const ACCENT: Color = Color::Indexed(214);

pub fn fg(color: Color) -> Style {
    Style::new().fg(color)
}

pub fn bold() -> Style {
    Style::new().add_modifier(Modifier::BOLD)
}

/// Python's `KIND_STYLE`; kinds it does not list are unstyled.
pub fn kind_style(kind: &str) -> Style {
    let c = |i: u8| fg(Color::Indexed(i));
    match kind {
        "LLM" => fg(MAGENTA).add_modifier(Modifier::BOLD),
        "decision" => c(170).add_modifier(Modifier::BOLD),
        "decision?" | "diffusion" => c(170),
        "embedding" => c(183),
        "speech" => c(153),
        "AI" => c(141),
        "compute" => c(172),
        "3D/game" => c(40),
        "video" => fg(YELLOW),
        "browser" => c(27),
        "app" => fg(CYAN),
        "compositor" | "kernel" | "system" | "external" => fg(GREY50),
        "wsl-vm" => c(47).add_modifier(Modifier::BOLD),
        _ => Style::new(),
    }
}

/// The VRAM bar's colour for a kind: its table colour, white when it has none.
pub fn bar_color(kind: &str) -> Color {
    kind_style(kind).fg.unwrap_or(WHITE)
}

/// Python's `SEG_STYLE` for the LLM breakdown bar.
pub fn segment_color(kind: SegmentKind) -> Color {
    match kind {
        SegmentKind::Weights => MAGENTA,
        SegmentKind::Kv => DARK_ORANGE,
        SegmentKind::Compute => YELLOW,
        SegmentKind::Runtime => GREY58,
        SegmentKind::Unknown => GREY42,
    }
}

/// One byte value as text: blank when the backend has no such counter, `n/a` when it has one but
/// this value is unknown, otherwise the size with what kind of number it is.
pub fn bytes_text(b: &Bytes) -> String {
    match *b {
        Bytes::Measured { bytes } => gb(bytes as f64),
        Bytes::Inferred { bytes, by } => format!("{}{}", gb(bytes as f64), by.marker()),
        Bytes::Estimated { bytes, .. } => format!("{} (est)", gb(bytes as f64)),
        Bytes::Reported { bytes, .. } => format!("{} (reported)", gb(bytes as f64)),
        Bytes::Unavailable {
            reason: Reason::NotOnBackend,
        } => String::new(),
        Bytes::Unavailable { .. } => "n/a".to_string(),
    }
}

/// A table cell: as `bytes_text`, but a measured value under `floor` is blank (Python hides tiny
/// spill and paged values).
pub fn bytes_cell(b: &Bytes, floor: u64) -> String {
    match *b {
        Bytes::Measured { bytes } if bytes < floor => String::new(),
        _ => bytes_text(b),
    }
}

/// A percentage cell: `n/a` (grey) when unknown, blank below 0.5, bold from 20.
pub fn pct_cell(v: Option<f64>) -> (String, Style) {
    match v {
        None => ("n/a".to_string(), fg(GREY62)),
        Some(v) if v < 0.5 => (String::new(), Style::new()),
        Some(v) => (
            format!("{v:.0}"),
            if v >= 20.0 { bold() } else { Style::new() },
        ),
    }
}

/// `4096` as `4,096` (Python's `{:,}`).
pub fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}
