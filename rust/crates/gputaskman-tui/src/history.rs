//! The eight trend charts and their per-GPU sample history (`app.py` `_record_sample` and
//! `chart_series`).
use gputaskman_core::collector::Snapshot;
use gputaskman_core::model::{DeviceTelemetry, VENDOR_AMD};
use ratatui::style::Color;

/// Samples kept per chart.
pub const CAPACITY: usize = 60;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Chart {
    Gpu,
    Clock,
    Vram,
    MemActivity,
    Temperature,
    Power,
    Encoder,
    Decoder,
}

/// Display order, as Python's `CHARTS`.
pub const CHARTS: [Chart; 8] = [
    Chart::Gpu,
    Chart::Clock,
    Chart::Vram,
    Chart::MemActivity,
    Chart::Temperature,
    Chart::Power,
    Chart::Encoder,
    Chart::Decoder,
];

impl Chart {
    /// The label in the chart grid; the clock chart says GFX on an AMD card.
    pub fn name(self, amd: bool) -> &'static str {
        match self {
            Chart::Gpu => "GPU",
            Chart::Clock if amd => "GFX clock",
            Chart::Clock => "SM clock",
            Chart::Vram => "VRAM",
            Chart::MemActivity => "Mem activity",
            Chart::Temperature => "Temperature",
            Chart::Power => "Power",
            Chart::Encoder => "Encoder",
            Chart::Decoder => "Decoder",
        }
    }

    /// The label in the chart selector, where one entry stands for every GPU's clock.
    pub fn menu_name(self, amd: bool, multi: bool) -> &'static str {
        if self == Chart::Clock && multi {
            "SM/GFX clock"
        } else {
            self.name(amd)
        }
    }

    /// Python's `CHART_COLORS`.
    pub fn color(self) -> Color {
        match self {
            Chart::Gpu => Color::Rgb(0x4a, 0xde, 0x80),
            Chart::Clock => Color::Rgb(0xfb, 0x92, 0x3c),
            Chart::Vram => Color::Rgb(0x60, 0xa5, 0xfa),
            Chart::MemActivity => Color::Rgb(0xa7, 0x8b, 0xfa),
            Chart::Temperature => Color::Rgb(0xef, 0x44, 0x44),
            Chart::Power => Color::Rgb(0xfa, 0xcc, 0x15),
            Chart::Encoder => Color::Rgb(0xe2, 0xe8, 0xf0),
            Chart::Decoder => Color::Rgb(0x22, 0xd3, 0xee),
        }
    }

    pub fn unit(self) -> &'static str {
        match self {
            Chart::Clock => " MHz",
            Chart::Temperature => "°C",
            Chart::Power => " W",
            _ => "%",
        }
    }
}

pub fn is_amd(s: &Snapshot) -> bool {
    s.adapter
        .as_ref()
        .is_some_and(|a| a.vendor_id == VENDOR_AMD)
}

/// One GPU's recent readings per chart. `None` is a missing reading (drawn blank), never 0.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct History {
    series: [Vec<Option<f64>>; 8],
}

impl History {
    pub fn record(&mut self, s: &Snapshot) {
        let amd = is_amd(s);
        let d = s.device.as_ref();
        let read = |f: fn(&DeviceTelemetry) -> Option<f64>| d.and_then(f);
        let clock = d
            .and_then(|d| if amd { d.gfx_clock } else { d.sm_clock })
            .map(f64::from);
        let vram = match (s.vram_used.value(), s.vram_total.value()) {
            (Some(used), Some(total)) if total > 0 => Some(100.0 * used as f64 / total as f64),
            _ => None,
        };
        let values = [
            read(|d| d.util_gpu),
            clock,
            vram,
            read(|d| d.util_mem),
            read(|d| d.temp),
            read(|d| d.power_w),
            read(|d| d.enc),
            read(|d| d.dec),
        ];
        for (series, v) in self.series.iter_mut().zip(values) {
            series.push(v);
            if series.len() > CAPACITY {
                series.remove(0);
            }
        }
    }

    pub fn values(&self, c: Chart) -> &[Option<f64>] {
        &self.series[c as usize]
    }

    pub fn current(&self, c: Chart) -> Option<f64> {
        self.values(c).last().copied().flatten()
    }

    /// The top of the chart: 100 for percentages, 120 for °C; for clock and power the larger of
    /// the device's maximum and the highest reading kept, at least 1.
    pub fn scale(&self, c: Chart, device: Option<&DeviceTelemetry>) -> f64 {
        let seen = self.values(c).iter().flatten().copied().fold(0.0, f64::max);
        match c {
            Chart::Clock => device
                .and_then(|d| d.sm_clock_max)
                .map_or(0.0, f64::from)
                .max(seen)
                .max(1.0),
            Chart::Power => device
                .and_then(|d| d.power_limit_w)
                .unwrap_or(0.0)
                .max(seen)
                .max(1.0),
            Chart::Temperature => 120.0,
            _ => 100.0,
        }
    }
}
