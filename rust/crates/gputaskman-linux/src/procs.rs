//! The `/proc` process source: names, command lines, start times, mapped files and listening ports.
//! Only metadata files are read (`stat`, `cmdline`, `exe`, `smaps`, `fd` links, `net/tcp*`); never
//! another process's memory, and never an equivalent of `psutil.open_files()`.
use gputaskman_core::model::Pid;
use gputaskman_core::provenance::Reason;
use gputaskman_core::traits::{MapEntry, ProcBasics, ProcessSource, StartTime};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub struct LinuxProcs {
    root: PathBuf,
    clk_tck: u64,
}

impl LinuxProcs {
    pub fn new(proc_root: PathBuf) -> Self {
        // SAFETY: `sysconf` takes a plain integer name and reads no memory of ours. This is the
        // crate's only `unsafe` block.
        let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
        let ticks = u64::try_from(ticks).ok().filter(|t| *t > 0).unwrap_or(100);
        Self::with_clk_tck(proc_root, ticks)
    }

    pub fn with_clk_tck(proc_root: PathBuf, ticks_per_sec: u64) -> Self {
        Self {
            root: proc_root,
            clk_tck: ticks_per_sec.max(1),
        }
    }

    fn read(&self, pid: Pid, file: &str) -> Option<Vec<u8>> {
        std::fs::read(self.root.join(pid.to_string()).join(file)).ok()
    }

    /// `(comm, fields after the closing parenthesis)`, split at the last `)` so names containing
    /// spaces or parentheses parse.
    fn stat(&self, pid: Pid) -> Option<(String, Vec<String>)> {
        let raw = self.read(pid, "stat")?;
        let text = String::from_utf8_lossy(&raw).into_owned();
        let open = text.find('(')?;
        let close = text.rfind(')')?;
        if close < open {
            return None;
        }
        let comm = text[open + 1..close].to_string();
        let rest = text[close + 1..]
            .split_whitespace()
            .map(str::to_string)
            .collect();
        Some((comm, rest))
    }

    fn btime(&self) -> Option<u64> {
        let text = std::fs::read_to_string(self.root.join("stat")).ok()?;
        text.lines()
            .find_map(|l| l.strip_prefix("btime "))
            .and_then(|v| v.trim().parse().ok())
    }

    fn start_from(&self, rest: &[String]) -> Option<StartTime> {
        // `rest` begins at field 3 (state), so field 22 is index 19.
        let ticks: u64 = rest.get(19)?.parse().ok()?;
        let wall = self.btime()? as f64 + ticks as f64 / self.clk_tck as f64;
        Some(StartTime { ticks, wall })
    }

    fn cmdline(&self, pid: Pid) -> Option<Vec<String>> {
        let raw = self.read(pid, "cmdline")?;
        let mut parts: Vec<String> = raw
            .split(|b| *b == 0)
            .map(|p| String::from_utf8_lossy(p).into_owned())
            .collect();
        while parts.last().is_some_and(String::is_empty) {
            parts.pop();
        }
        Some(parts)
    }
}

/// psutil: `comm` is truncated by the kernel at 15 characters, so a longer cmdline basename that
/// starts with it is the real name.
fn display_name(comm: String, cmdline: Option<&Vec<String>>) -> String {
    if comm.chars().count() == 15
        && let Some(first) = cmdline.and_then(|c| c.first())
        && let Some(base) = Path::new(first).file_name().and_then(|b| b.to_str())
        && base.starts_with(&comm)
    {
        return base.to_string();
    }
    comm
}

impl ProcessSource for LinuxProcs {
    fn pids(&self) -> Vec<Pid> {
        let Ok(read) = std::fs::read_dir(&self.root) else {
            return Vec::new();
        };
        let mut pids: Vec<Pid> = read
            .filter_map(Result::ok)
            .filter_map(|e| e.file_name().to_str()?.parse().ok())
            .collect();
        pids.sort_unstable();
        pids
    }

    fn basics(&self, pid: Pid) -> Option<ProcBasics> {
        let (comm, rest) = self.stat(pid)?;
        let cmdline = self.cmdline(pid);
        let exe = std::fs::read_link(self.root.join(pid.to_string()).join("exe"))
            .ok()
            .map(|p| p.to_string_lossy().into_owned());
        Some(ProcBasics {
            name: Some(display_name(comm, cmdline.as_ref())),
            exe,
            cmdline,
            start: self.start_from(&rest),
        })
    }

    fn start_time(&self, pid: Pid) -> Option<StartTime> {
        let (_, rest) = self.stat(pid)?;
        self.start_from(&rest)
    }

    fn parent(&self, pid: Pid) -> Option<Pid> {
        let (_, rest) = self.stat(pid)?;
        rest.get(1)?.parse().ok()
    }

    fn memory_maps(&self, pid: Pid) -> Result<Vec<MapEntry>, Reason> {
        let raw = self.read(pid, "smaps").ok_or(Reason::ProcessUnreadable)?;
        let text = String::from_utf8_lossy(&raw);
        let mut order: Vec<String> = Vec::new();
        let mut sums: BTreeMap<String, u64> = BTreeMap::new();
        let mut current: Option<String> = None;
        for line in text.lines() {
            if let Some(kb) = line.strip_prefix("Rss:") {
                let kb: u64 = kb
                    .split_whitespace()
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
                if let Some(path) = &current {
                    *sums.entry(path.clone()).or_insert(0) += kb * 1024;
                }
            } else if is_header(line) {
                // `address perms offset dev inode [pathname]`
                let path = pathname(line);
                current = path;
                if let Some(p) = &current
                    && !sums.contains_key(p)
                {
                    order.push(p.clone());
                    sums.insert(p.clone(), 0);
                }
            }
        }
        Ok(order
            .into_iter()
            .map(|path| {
                let rss = sums.get(&path).copied().unwrap_or(0);
                MapEntry { path, rss }
            })
            .collect())
    }

    fn listening_ports(&self) -> BTreeMap<Pid, Vec<u16>> {
        let mut by_inode: BTreeMap<u64, u16> = BTreeMap::new();
        for table in ["tcp", "tcp6"] {
            let Ok(text) = std::fs::read_to_string(self.root.join("net").join(table)) else {
                continue;
            };
            for line in text.lines().skip(1) {
                let f: Vec<&str> = line.split_whitespace().collect();
                if f.len() < 10 || f[3] != "0A" {
                    continue;
                }
                let port = f[1]
                    .rsplit(':')
                    .next()
                    .and_then(|p| u16::from_str_radix(p, 16).ok());
                if let (Some(port), Ok(inode)) = (port, f[9].parse::<u64>()) {
                    by_inode.insert(inode, port);
                }
            }
        }
        let mut found: BTreeMap<Pid, BTreeSet<u16>> = BTreeMap::new();
        if by_inode.is_empty() {
            return BTreeMap::new();
        }
        for pid in self.pids() {
            // Other users' fd directories are unreadable: skipped, not an error.
            let Ok(fds) = std::fs::read_dir(self.root.join(pid.to_string()).join("fd")) else {
                continue;
            };
            for fd in fds.filter_map(Result::ok) {
                let Ok(target) = std::fs::read_link(fd.path()) else {
                    continue;
                };
                let target = target.to_string_lossy();
                let inode = target
                    .strip_prefix("socket:[")
                    .and_then(|t| t.strip_suffix(']'))
                    .and_then(|t| t.parse::<u64>().ok());
                if let Some(port) = inode.and_then(|i| by_inode.get(&i)) {
                    found.entry(pid).or_default().insert(*port);
                }
            }
        }
        found
            .into_iter()
            .map(|(pid, ports)| (pid, ports.into_iter().collect()))
            .collect()
    }

    fn self_pid(&self) -> Pid {
        std::process::id()
    }
}

/// A smaps mapping header starts with `<hex>-<hex> `; field lines start with a capitalised key.
fn is_header(line: &str) -> bool {
    let Some(first) = line.split_whitespace().next() else {
        return false;
    };
    match first.split_once('-') {
        Some((a, b)) => {
            !a.is_empty()
                && !b.is_empty()
                && a.chars().all(|c| c.is_ascii_hexdigit())
                && b.chars().all(|c| c.is_ascii_hexdigit())
        }
        None => false,
    }
}

/// The pathname after the fifth whitespace-separated field (padded with spaces, and it may itself
/// contain spaces). `None` for anonymous mappings.
fn pathname(line: &str) -> Option<String> {
    let mut rest = line;
    for _ in 0..5 {
        rest = rest.trim_start();
        rest = &rest[rest.find(char::is_whitespace)?..];
    }
    let path = rest.trim();
    (!path.is_empty()).then(|| path.to_string())
}
