//! Read-only helpers for sysfs and procfs files (`drm.py` `_read` and `_int`).
use std::path::Path;

/// Contents of a sysfs/procfs file, trimmed; `None` when it is missing or unreadable.
pub fn read_trimmed(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    Some(String::from_utf8_lossy(&bytes).trim().to_string())
}

/// Integer in a sysfs file; `None` when missing, empty or unparseable (unknown, not 0).
/// A `0x` prefix is accepted for radix 16, as Python's `int(value, 16)` does.
pub fn read_int(path: &Path, radix: u32) -> Option<i64> {
    let text = read_trimmed(path)?;
    if text.is_empty() {
        return None;
    }
    let digits = if radix == 16 {
        text.strip_prefix("0x")
            .or_else(|| text.strip_prefix("0X"))
            .unwrap_or(&text)
    } else {
        &text
    };
    i64::from_str_radix(digits, radix).ok()
}
