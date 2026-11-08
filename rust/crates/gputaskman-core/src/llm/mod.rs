//! What is inside an LLM process's VRAM: flags, load logs and local server probes.
pub mod flags;
pub mod logs;
pub mod servers;
pub mod view;

pub(crate) const MIB: u64 = 1 << 20;

/// Human size. Rounding is half-to-even, as Python's format does.
pub fn gb(n: f64) -> String {
    if n.abs() >= (1u64 << 30) as f64 {
        format!("{:.2} GB", n / (1u64 << 30) as f64)
    } else {
        format!("{:.0} MB", n / MIB as f64)
    }
}
