//! Device metrics for one AMD card from sysfs and hwmon (`drm.py` `read_device`).
use crate::drm::cards::DrmCard;
use crate::sysfs::{read_int, read_trimmed};
use gputaskman_core::model::DeviceTelemetry;
use std::path::{Path, PathBuf};

fn sorted_names(dir: &Path) -> Vec<String> {
    let Ok(read) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = read
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

fn hwmon_dir(device: &Path) -> Option<PathBuf> {
    let hwmon = device.join("hwmon");
    sorted_names(&hwmon)
        .into_iter()
        .next()
        .map(|n| hwmon.join(n))
}

fn int(path: &Path) -> Option<i64> {
    read_int(path, 10)
}

fn unsigned(path: &Path) -> Option<u64> {
    int(path).and_then(|v| u64::try_from(v).ok())
}

fn percent(path: &Path) -> Option<f64> {
    int(path).map(|v| v as f64)
}

/// Python floor division by 1000, clamped to the unsigned fields it fills.
fn milli(path: &Path) -> Option<i64> {
    int(path).map(|v| v.div_euclid(1000))
}

fn microwatts(path: &Path) -> Option<f64> {
    int(path).map(|v| v as f64 / 1_000_000.0)
}

fn hwmon_temp(hwmon: &Path) -> Option<i64> {
    for name in sorted_names(hwmon) {
        if let Some(stem) = name.strip_suffix("_label")
            && name.starts_with("temp")
            && read_trimmed(&hwmon.join(&name)).as_deref() == Some("edge")
        {
            return milli(&hwmon.join(format!("{stem}_input")));
        }
    }
    milli(&hwmon.join("temp1_input"))
}

fn hwmon_clock(hwmon: &Path, label: &str) -> Option<u32> {
    for name in sorted_names(hwmon) {
        if let Some(stem) = name.strip_suffix("_label")
            && name.starts_with("freq")
            && read_trimmed(&hwmon.join(&name)).as_deref() == Some(label)
        {
            let value = int(&hwmon.join(format!("{stem}_input")))?;
            return u32::try_from(value.div_euclid(1_000_000)).ok();
        }
    }
    None
}

fn pcie_gen(speed: &str) -> Option<u32> {
    let value: f64 = speed.split_whitespace().next()?.parse().ok()?;
    match (value * 10.0).round_ties_even() as i64 {
        25 => Some(1),
        50 => Some(2),
        80 => Some(3),
        160 => Some(4),
        320 => Some(5),
        640 => Some(6),
        _ => None,
    }
}

/// `MHz` values from `pp_dpm_sclk` lines such as `2: 2570Mhz *`.
fn mhz_values(text: &str) -> Vec<u32> {
    let lower = text.to_lowercase();
    let bytes = lower.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            let digits = &lower[start..i];
            let rest = lower[i..].trim_start();
            if rest.starts_with("mhz")
                && let Ok(v) = digits.parse()
            {
                out.push(v);
            }
        } else {
            i += 1;
        }
    }
    out
}

/// NVIDIA-shaped device metrics for one AMD card, read from sysfs. The caller fills the driver
/// version from `kernel_release` when it wants it shown; `driver_model` is the kernel driver name.
pub fn read_device(card: &DrmCard, index: usize, _kernel_release: &str) -> DeviceTelemetry {
    let d = &card.sysfs;
    let mut dev = DeviceTelemetry {
        index: u32::try_from(index).unwrap_or(u32::MAX),
        name: card.name.clone(),
        pci_bus: card.pdev.clone(),
        driver_model: card.driver.clone(),
        source: "AMD sysfs".to_string(),
        mem_total: unsigned(&d.join("mem_info_vram_total")),
        mem_used: unsigned(&d.join("mem_info_vram_used")),
        util_gpu: percent(&d.join("gpu_busy_percent")),
        util_mem: percent(&d.join("mem_busy_percent")),
        ..DeviceTelemetry::default()
    };
    if let Some(hwmon) = hwmon_dir(d) {
        dev.temp = hwmon_temp(&hwmon).map(|v| v as f64);
        let mut power = microwatts(&hwmon.join("power1_average"));
        if power.is_none_or(|p| p == 0.0) {
            // some parts expose power1_average but leave it at 0
            if let Some(input) = microwatts(&hwmon.join("power1_input")) {
                power = Some(input);
            }
        }
        dev.power_w = power;
        if power.is_some() {
            dev.power_kind = read_trimmed(&hwmon.join("power1_label")).unwrap_or_default();
        }
        dev.power_limit_w = microwatts(&hwmon.join("power1_cap"));
        if let (Some(input), Some(max)) =
            (int(&hwmon.join("fan1_input")), int(&hwmon.join("fan1_max")))
            && max > 0
        {
            dev.fan =
                u32::try_from((100.0 * input as f64 / max as f64).round_ties_even() as i64).ok();
        }
        dev.gfx_clock = hwmon_clock(&hwmon, "sclk");
        dev.mem_clock = hwmon_clock(&hwmon, "mclk");
    }
    let text = read_trimmed(&d.join("pp_dpm_sclk")).unwrap_or_default();
    dev.sm_clock_max = mhz_values(&text).into_iter().max();
    if let Some(speed) = read_trimmed(&d.join("current_link_speed")) {
        dev.pcie_gen = pcie_gen(&speed);
    }
    dev.pcie_width = int(&d.join("current_link_width")).and_then(|v| u32::try_from(v).ok());
    dev
}

/// Major version of the card's video block (VCN, listed as UVD in IP discovery); 0 if unknown.
pub fn vcn_major(device: &Path) -> i64 {
    int(&device.join("ip_discovery/die/0/UVD/0/major")).unwrap_or(0)
}

pub fn engine_key(name: &str, unified_video: bool) -> String {
    let lowered = name.to_lowercase();
    let has = |s: &str| lowered.contains(s);
    let key = if unified_video && (has("enc") || has("dec")) {
        "video" // one ring for both: decode and encode cannot be told apart
    } else if lowered == "gfx" {
        "3d"
    } else if lowered == "compute" {
        "compute"
    } else if lowered == "enc" {
        "encode"
    } else if lowered == "dec" || has("dec") || has("jpeg") {
        "decode"
    } else if has("enc") {
        "encode"
    } else if lowered == "dma" {
        "copy" // SDMA: buffer copies and uploads
    } else {
        "other" // vpe, ...: busy time still counts toward GPU%, like WDDM's unnamed engines
    };
    key.to_string()
}
