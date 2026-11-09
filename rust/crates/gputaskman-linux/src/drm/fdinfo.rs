//! Parsing of one `/proc/<pid>/fdinfo/<fd>` file (`drm.py` `parse_size` and `parse_fdinfo`).
use std::collections::{BTreeMap, HashMap};

/// `'2060 KiB'` to bytes; a bare number is bytes; anything unparseable is `None` (unknown, not 0).
pub fn parse_size(value: &str) -> Option<u64> {
    let mut parts = value.split_whitespace();
    let number: u64 = parts.next()?.parse().ok()?;
    let unit = parts.next().map(str::to_lowercase).unwrap_or_default();
    let multiplier: u64 = match unit.as_str() {
        "" | "b" => 1,
        "kb" => 1000,
        "kib" => 1024,
        "mb" => 1000 * 1000,
        "mib" => 1024 * 1024,
        "gb" => 1000 * 1000 * 1000,
        "gib" => 1024 * 1024 * 1024,
        "tb" => 1000u64.pow(4),
        "tib" => 1024u64.pow(4),
        _ => return None,
    };
    number.checked_mul(multiplier)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FdInfo {
    pub driver: String,
    pub client_id: Option<u64>,
    pub pdev: String,
    pub vram: Option<u64>,
    pub gtt: Option<u64>,
    pub shared: Option<u64>,
    pub cpu: Option<u64>,
    /// `None` when any engine counter is torn or malformed: the client's load is unknown, not 0.
    pub engines: Option<BTreeMap<String, u64>>,
}

/// One fdinfo file; `None` when it is not a DRM fd.
pub fn parse_fdinfo(text: &str) -> Option<FdInfo> {
    let mut fields: HashMap<&str, &str> = HashMap::new();
    for line in text.lines() {
        if let Some((key, value)) = line.split_once(':') {
            fields.insert(key.trim(), value.trim());
        }
    }
    let driver = (*fields.get("drm-driver")?).to_string();
    let client_id = fields
        .get("drm-client-id")
        .and_then(|v| v.parse::<u64>().ok());

    let memory = |resident: &str, legacy: &str| {
        fields
            .get(resident)
            .or_else(|| fields.get(legacy))
            .and_then(|v| parse_size(v))
    };

    // drm-shared-<region> counts buffers with more than one handle by their preferred region, while
    // drm-resident-<region> counts by where buffers are now, so the two cannot be subtracted. Every
    // standard shared region, including CPU, must be present and parseable, or shared is unknown.
    let required = ["drm-shared-vram", "drm-shared-gtt", "drm-shared-cpu"];
    let shared_values: Vec<Option<u64>> = fields
        .iter()
        .filter(|(k, _)| k.starts_with("drm-shared-"))
        .map(|(_, v)| parse_size(v))
        .collect();
    let shared = if required.iter().all(|k| fields.contains_key(k))
        && !shared_values.is_empty()
        && shared_values.iter().all(Option::is_some)
    {
        shared_values
            .iter()
            .flatten()
            .try_fold(0u64, |a, b| a.checked_add(*b))
    } else {
        None
    };

    let mut engines = Some(BTreeMap::new());
    for (key, value) in &fields {
        let Some(name) = key.strip_prefix("drm-engine-") else {
            continue;
        };
        if key.starts_with("drm-engine-capacity-") {
            continue;
        }
        match value
            .split_whitespace()
            .next()
            .and_then(|v| v.parse::<u64>().ok())
        {
            Some(ns) => {
                if let Some(map) = engines.as_mut() {
                    map.insert(name.to_string(), ns);
                }
            }
            None => {
                engines = None;
                break;
            }
        }
    }

    Some(FdInfo {
        driver,
        client_id,
        pdev: fields.get("drm-pdev").copied().unwrap_or("").to_string(),
        vram: memory("drm-resident-vram", "drm-memory-vram"),
        gtt: memory("drm-resident-gtt", "drm-memory-gtt"),
        shared,
        cpu: memory("drm-resident-cpu", "drm-memory-cpu"),
        engines,
    })
}
