//! amdgpu cards from `/sys/class/drm` (`drm.py` `cards` and `adapters`).
use crate::drm::device::vcn_major;
use crate::sysfs::{read_int, read_trimmed};
use gputaskman_core::model::{Adapter, BackendKind, pci_key};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

pub const DRIVERS: &[&str] = &["amdgpu"];
/// hwdata (Arch, Fedora) and Debian/Ubuntu locations.
pub const PCI_IDS: &[&str] = &["/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrmCard {
    pub key: u64,
    pub pdev: String,
    pub sysfs: PathBuf,
    pub driver: String,
    pub vendor_id: u16,
    pub device_id: u16,
    pub name: String,
    pub vram_total: u64,
    pub integrated: bool,
    /// VCN 4+ runs decode and encode on one ring that fdinfo reports as "enc".
    pub unified_video: bool,
}

fn uevent(device: &Path) -> HashMap<String, String> {
    let text = read_trimmed(&device.join("uevent")).unwrap_or_default();
    text.lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect()
}

fn driver_of(device: &Path) -> String {
    match std::fs::read_link(device.join("driver")) {
        Ok(target) => target
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        Err(_) => uevent(device).get("DRIVER").cloned().unwrap_or_default(),
    }
}

fn hex_id(path: &Path) -> u16 {
    read_int(path, 16)
        .and_then(|v| u16::try_from(v).ok())
        .unwrap_or(0)
}

/// `{vendor id: {device id: name}}` from pci.ids; subsystem lines (two tabs) are skipped.
fn pci_ids(path: &Path) -> BTreeMap<u16, BTreeMap<u16, String>> {
    let mut out: BTreeMap<u16, BTreeMap<u16, String>> = BTreeMap::new();
    let Ok(bytes) = std::fs::read(path) else {
        return out;
    };
    let text = String::from_utf8_lossy(&bytes);
    let mut vendor: Option<u16> = None;
    for line in text.lines() {
        if line.trim().is_empty() || line.starts_with('#') || line.starts_with("\t\t") {
            continue;
        }
        if line.starts_with('\t') {
            let Some(v) = vendor else { continue };
            let Some((id, name)) = split_first_word(line.trim()) else {
                continue;
            };
            if let Ok(device) = u16::from_str_radix(id, 16) {
                out.entry(v)
                    .or_default()
                    .insert(device, name.trim().to_string());
            }
            continue;
        }
        vendor = split_first_word(line).and_then(|(id, _)| u16::from_str_radix(id, 16).ok());
    }
    out
}

fn split_first_word(s: &str) -> Option<(&str, &str)> {
    let s = s.trim_start();
    let end = s.find(char::is_whitespace)?;
    let rest = s[end..].trim_start();
    if rest.is_empty() {
        return None;
    }
    Some((&s[..end], rest))
}

fn vendor_prefix(vendor_id: u16) -> String {
    match vendor_id {
        0x10de => "NVIDIA".into(),
        0x1002 => "AMD".into(),
        0x8086 => "Intel".into(),
        0x1414 => "Microsoft".into(),
        0x5143 => "Qualcomm".into(),
        other => format!("{other:#x}"),
    }
}

fn card_name(device: &Path, vendor_id: u16, device_id: u16, pci_ids_path: Option<&Path>) -> String {
    if let Some(product) = read_trimmed(&device.join("product_name"))
        && !product.is_empty()
    {
        return product;
    }
    let prefix = vendor_prefix(vendor_id);
    let paths: Vec<PathBuf> = match pci_ids_path {
        Some(p) => vec![p.to_path_buf()],
        None => PCI_IDS.iter().map(PathBuf::from).collect(),
    };
    let entry = paths.iter().find_map(|p| {
        pci_ids(p)
            .get(&vendor_id)
            .and_then(|m| m.get(&device_id))
            .filter(|e| !e.is_empty())
            .cloned()
    });
    let Some(entry) = entry else {
        return format!("{prefix} GPU {device_id:04x}");
    };
    if let (Some(open), Some(close)) = (entry.find('['), entry.rfind(']'))
        && open < close
    {
        let inner = &entry[open + 1..close];
        if !inner.is_empty() {
            return format!("{prefix} {inner}");
        }
    }
    format!("{prefix} {entry}")
}

fn is_card_name(name: &str) -> bool {
    name.strip_prefix("card")
        .is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
}

/// DRM cards backed by an amdgpu PCI device, discrete first then by VRAM descending.
pub fn cards(root: &Path, pci_ids: Option<&Path>) -> Vec<DrmCard> {
    let Ok(read) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut entries: Vec<(String, PathBuf)> = read
        .filter_map(Result::ok)
        .map(|e| (e.file_name().to_string_lossy().into_owned(), e.path()))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut found = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for (name, path) in entries {
        if !is_card_name(&name) {
            continue;
        }
        let device = path.join("device");
        let driver = driver_of(&device);
        if !DRIVERS.contains(&driver.as_str()) {
            continue;
        }
        let pdev = uevent(&device)
            .get("PCI_SLOT_NAME")
            .filter(|s| !s.is_empty())
            .cloned()
            .or_else(|| {
                std::fs::canonicalize(&device)
                    .ok()
                    .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            })
            .unwrap_or_default();
        if !seen.insert(pdev.clone()) {
            continue;
        }
        let vendor_id = hex_id(&device.join("vendor"));
        let device_id = hex_id(&device.join("device"));
        let vram_total = read_int(&device.join("mem_info_vram_total"), 10)
            .and_then(|v| u64::try_from(v).ok())
            .unwrap_or(0);
        found.push(DrmCard {
            key: pci_key(&pdev),
            name: card_name(&device, vendor_id, device_id, pci_ids),
            pdev,
            driver,
            vendor_id,
            device_id,
            vram_total,
            // amdgpu reports the VRAM chips' vendor only for a discrete card; an APU's "VRAM" is a carve-out.
            integrated: !device.join("mem_info_vram_vendor").exists(),
            unified_video: vcn_major(&device) >= 4,
            sysfs: device,
        });
    }
    found.sort_by_key(|c| (c.integrated, std::cmp::Reverse(c.vram_total)));
    found
}

/// AMD cards as bare adapters (per-process data comes from the DRM counters).
pub fn adapters(found: &[DrmCard]) -> Vec<Adapter> {
    found
        .iter()
        .map(|c| Adapter {
            key: c.key,
            pci_bus: c.pdev.clone(),
            name: c.name.clone(),
            vendor_id: c.vendor_id,
            dedicated_vram: c.vram_total,
            shared_system: read_int(&c.sysfs.join("mem_info_gtt_total"), 10)
                .and_then(|v| u64::try_from(v).ok())
                .unwrap_or(0),
            integrated: c.integrated,
            backend: BackendKind::Drm,
        })
        .collect()
}
