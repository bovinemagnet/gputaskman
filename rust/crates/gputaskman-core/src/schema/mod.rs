//! Versioned JSON output, and readers for saved snapshots.
use std::path::Path;

use serde_json::Value;

use self::v2::SnapshotV2;

pub mod v1;
pub mod v2;

pub const SCHEMA_VERSION: u32 = 2;

/// Decode a saved snapshot: UTF-16 with a byte order mark (a Windows PowerShell 5.1 redirect), or
/// UTF-8 with or without one.
fn decode(raw: &[u8]) -> Result<String, String> {
    let utf16 = |bytes: &[u8], le: bool| {
        let units: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|c| {
                if le {
                    u16::from_le_bytes([c[0], c[1]])
                } else {
                    u16::from_be_bytes([c[0], c[1]])
                }
            })
            .collect();
        String::from_utf16(&units).map_err(|e| e.to_string())
    };
    match raw {
        [0xff, 0xfe, rest @ ..] => utf16(rest, true),
        [0xfe, 0xff, rest @ ..] => utf16(rest, false),
        [0xef, 0xbb, 0xbf, rest @ ..] => {
            String::from_utf8(rest.to_vec()).map_err(|e| e.to_string())
        }
        _ => String::from_utf8(raw.to_vec()).map_err(|e| e.to_string()),
    }
}

/// Read a saved `--json` snapshot: v2 when `schema_version` is 2, else a Python v1 document.
pub fn load(path: &Path) -> Result<SnapshotV2, String> {
    let shown = path.display();
    let raw = std::fs::read(path).map_err(|e| format!("{shown}: {e}"))?;
    let text = decode(&raw).map_err(|e| format!("{shown}: {e}"))?;
    let data: Value = serde_json::from_str(&text).map_err(|e| format!("{shown}: {e}"))?;
    if !data.get("processes").is_some_and(Value::is_array) {
        return Err(format!("{shown} is not a gputaskman --json snapshot"));
    }
    if data.get("schema_version").and_then(Value::as_u64) == Some(u64::from(SCHEMA_VERSION)) {
        serde_json::from_value(data).map_err(|e| format!("{shown}: {e}"))
    } else {
        v1::from_v1(&data)
    }
}
