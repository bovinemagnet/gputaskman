//! Per-process VRAM and engine load from DRM fdinfo (`drm.py` `DrmCounters`).
//!
//! A process is charged bytes only when every DRM client it holds is provably private: one visible
//! holder, reached through a render node, no shared buffers at all, and every value known. Resident
//! bytes are kept as a reference and never summed (AGENTS.md rules 1 to 3).
use crate::drm::cards::{DRIVERS, DrmCard};
use crate::drm::device::engine_key;
use crate::drm::fdinfo::{FdInfo, parse_fdinfo};
use crate::sysfs::read_int;
use gputaskman_core::model::{AdapterGpu, Engines, Pid, ProcGpu};
use gputaskman_core::provenance::{Bytes, Reason, Resident};
use gputaskman_core::traits::Clock;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// Time per refresh for listing process fds (one-shot output: no limit).
pub const SCAN_BUDGET: Duration = Duration::from_millis(50);
/// Forget engine baselines for clients not seen for a minute.
const PRUNE_NS: u128 = 60 * 1_000_000_000;

/// One DRM client: `(pdev, client id)`, or `(pdev, pid, fd)` when fdinfo gives no ID.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ClientKey {
    Id(String, u64),
    Fd(String, Pid, String),
}

/// What a scan of one process's fds found.
#[derive(Debug, Clone)]
struct Known {
    /// Field 22 of `stat`: tells a reused pid from the process cached under it.
    start: Option<String>,
    /// `(fd name, link target)` for each `/dev/dri` fd.
    dri: Vec<(String, String)>,
    holds_kfd: bool,
    /// Every fd link was readable.
    complete: bool,
}

enum Scan {
    /// Its fds can't be listed.
    Denied,
    /// The budget ran out partway through its fds.
    Deferred,
    Found(Known),
}

struct Client {
    owners: BTreeSet<Pid>,
    info: FdInfo,
    /// Every holder reached it through a render node.
    render: bool,
    read_at: u128,
}

struct Clients {
    /// In discovery order, as Python's dict keeps them.
    list: Vec<(ClientKey, Client)>,
    kfd: BTreeSet<Pid>,
    incomplete: BTreeSet<Pid>,
}

/// Same shape as the WDDM counters: `sample()` gives `{adapter key: {pid: ProcGpu}}` and
/// `{adapter key: AdapterGpu}`. Call `start()` once the card is selected: engine load is a rate, so
/// it takes the first (unbudgeted) baseline.
pub struct DrmCounters {
    cards: Vec<DrmCard>,
    proc_root: PathBuf,
    clock: Arc<dyn Clock>,
    /// Client to (high-water engine ns, read time).
    prev: HashMap<ClientKey, (BTreeMap<String, u64>, u128)>,
    known: BTreeMap<Pid, Known>,
    cursor: Pid,
    status: String,
    partial: bool,
    budget: Option<Duration>,
}

fn read_lossy(path: &Path) -> std::io::Result<String> {
    std::fs::read(path).map(|b| String::from_utf8_lossy(&b).into_owned())
}

impl DrmCounters {
    /// Takes no sample: an NVIDIA GPU may be selected, so no fd walk happens until `start()`.
    pub fn new(cards: Vec<DrmCard>, proc_root: PathBuf, clock: Arc<dyn Clock>) -> Self {
        Self {
            cards,
            proc_root,
            clock,
            prev: HashMap::new(),
            known: BTreeMap::new(),
            cursor: 0,
            status: String::new(),
            partial: false,
            budget: Some(SCAN_BUDGET),
        }
    }

    /// Take the engine baseline over every process, like the WDDM counters do at startup.
    pub fn start(&mut self) {
        if self.cards.is_empty() {
            return;
        }
        let budget = self.budget.take();
        self.sample();
        self.budget = budget;
    }

    /// `None` lists every process on each refresh.
    pub fn set_budget(&mut self, budget: Option<Duration>) {
        self.budget = budget;
    }

    /// What the last scan couldn't cover, for the UI and JSON; empty when it saw everything.
    pub fn status(&self) -> &str {
        &self.status
    }

    /// The last scan ran out of budget, so a DRM client's holders may not all be known.
    pub fn partial(&self) -> bool {
        self.partial
    }

    /// Processes whose fds are cached from the last scan.
    #[cfg(feature = "test-support")]
    pub fn known_pids(&self) -> Vec<Pid> {
        self.known.keys().copied().collect()
    }

    /// Where the rotation resumes on the next budgeted scan.
    #[cfg(feature = "test-support")]
    pub fn cursor(&self) -> Pid {
        self.cursor
    }

    #[cfg(feature = "test-support")]
    pub fn set_cursor(&mut self, pid: Pid) {
        self.cursor = pid;
    }

    fn now_ns(&self) -> u128 {
        self.clock.mono().as_nanos()
    }

    fn adapters(&self) -> BTreeMap<u64, AdapterGpu> {
        let used = |path: PathBuf| match read_int(&path, 10).and_then(|v| u64::try_from(v).ok()) {
            Some(v) => Bytes::measured(v),
            None => Bytes::unavailable(Reason::NotOnBackend), // unknown, not 0
        };
        self.cards
            .iter()
            .map(|c| {
                let gpu = AdapterGpu {
                    vram_used: used(c.sysfs.join("mem_info_vram_used")),
                    shared_used: used(c.sysfs.join("mem_info_gtt_used")),
                };
                (c.key, gpu)
            })
            .collect()
    }

    /// Field 22 of `/proc/<pid>/stat`.
    fn start_time(&self, pid: Pid) -> Option<String> {
        let text = read_lossy(&self.proc_root.join(pid.to_string()).join("stat")).ok()?;
        let (_, rest) = text.rsplit_once(')')?;
        rest.split_whitespace().nth(19).map(str::to_string)
    }

    fn scan_pid(&self, pid: Pid, deadline: Option<Duration>) -> Scan {
        let start = self.start_time(pid);
        let mut found = Known {
            start,
            dri: Vec::new(),
            holds_kfd: false,
            complete: true,
        };
        let fds = match std::fs::read_dir(self.proc_root.join(pid.to_string()).join("fd")) {
            Ok(fds) => fds,
            Err(e) if e.kind() == ErrorKind::PermissionDenied => return Scan::Denied,
            Err(_) => return Scan::Found(found), // gone since /proc was listed
        };
        for (n, fd) in fds.enumerate().map(|(i, fd)| (i + 1, fd)) {
            if let Some(at) = deadline
                && n % 64 == 0
                && self.clock.mono() >= at
            {
                return Scan::Deferred;
            }
            let Ok(fd) = fd else { continue };
            let target = match std::fs::read_link(fd.path()) {
                Ok(t) => t.to_string_lossy().into_owned(),
                Err(e) if e.kind() == ErrorKind::PermissionDenied => {
                    found.complete = false;
                    continue;
                }
                Err(_) => continue, // closed since the listing
            };
            if target == "/dev/kfd" {
                found.holds_kfd = true;
            } else if target.starts_with("/dev/dri/") {
                found
                    .dri
                    .push((fd.file_name().to_string_lossy().into_owned(), target));
            }
        }
        Scan::Found(found)
    }

    fn pids(&self) -> std::io::Result<Vec<Pid>> {
        let mut pids: Vec<Pid> = std::fs::read_dir(&self.proc_root)?
            .filter_map(Result::ok)
            .filter_map(|e| {
                let name = e.file_name();
                let name = name.to_str()?;
                if name.is_empty() || !name.bytes().all(|b| b.is_ascii_digit()) {
                    return None;
                }
                name.parse().ok()
            })
            .collect();
        pids.sort_unstable();
        Ok(pids)
    }

    /// One entry per DRM client, plus the pids holding `/dev/kfd` and the pids whose fds or fdinfo
    /// couldn't all be read.
    ///
    /// Listing a process's fds is the costly part, so it is bounded by the budget: processes not
    /// seen before go first, then the rest resume where the last refresh stopped. Past the budget a
    /// process keeps its last known fds (dropped if its pid was reused), whose fdinfo is still read:
    /// rows and counters stay current, and only an fd handed to an already-running process shows up
    /// a refresh or two late.
    fn clients(&mut self) -> Clients {
        let by_pdev: BTreeSet<&str> = self.cards.iter().map(|c| c.pdev.as_str()).collect();
        let mut out = Clients {
            list: Vec::new(),
            kfd: BTreeSet::new(),
            incomplete: BTreeSet::new(),
        };
        let mut index: HashMap<ClientKey, usize> = HashMap::new();
        self.status.clear();
        self.partial = false;
        let Ok(pids) = self.pids() else {
            self.status = format!("{} is not readable", self.proc_root.display());
            return out;
        };
        let start = pids.partition_point(|&p| p < self.cursor);
        let rotated: Vec<Pid> = pids[start..]
            .iter()
            .chain(&pids[..start])
            .copied()
            .collect();
        let ordered: Vec<Pid> = rotated
            .iter()
            .filter(|p| !self.known.contains_key(p))
            .chain(rotated.iter().filter(|p| self.known.contains_key(p)))
            .copied()
            .collect();
        let deadline = self.budget.map(|b| self.clock.mono() + b);
        let mut known: BTreeMap<Pid, Known> = BTreeMap::new();
        let (mut denied, mut deferred) = (0usize, 0usize);
        let mut out_of_time = false;
        // First known process not rescanned: where the rotation resumes (new ones go first anyway).
        let mut cursor: Option<Pid> = None;
        let mut scanned = 0usize;
        for pid in ordered {
            let mut entry: Option<Known> = None;
            if !out_of_time && deadline.is_none_or(|at| self.clock.mono() < at) {
                // The first process of a refresh is always listed in full, so one huge fd table
                // can't stall the rotation.
                let scan = self.scan_pid(pid, if scanned > 0 { deadline } else { None });
                scanned += 1;
                match scan {
                    Scan::Denied => {
                        denied += 1;
                        continue;
                    }
                    Scan::Deferred => {}
                    Scan::Found(k) => entry = Some(k),
                }
            }
            let entry = match entry {
                Some(e) => e,
                None => {
                    // Not scanned this refresh: reuse the cached fds if this is still the same process.
                    out_of_time = true;
                    if cursor.is_none() && self.known.contains_key(&pid) {
                        cursor = Some(pid);
                    }
                    deferred += 1;
                    match self.known.get(&pid) {
                        Some(cached) if cached.start == self.start_time(pid) => cached.clone(),
                        _ => continue,
                    }
                }
            };
            if entry.holds_kfd {
                out.kfd.insert(pid);
            }
            if !entry.complete {
                out.incomplete.insert(pid);
            }
            for (name, target) in &entry.dri {
                let path = self
                    .proc_root
                    .join(pid.to_string())
                    .join("fdinfo")
                    .join(name);
                let text = match read_lossy(&path) {
                    Ok(t) => t,
                    Err(e) if e.kind() == ErrorKind::PermissionDenied => {
                        out.incomplete.insert(pid);
                        continue;
                    }
                    Err(_) => continue, // closed since the listing
                };
                let read_at = self.now_ns();
                let Some(info) = parse_fdinfo(&text) else {
                    continue;
                };
                if !DRIVERS.contains(&info.driver.as_str()) || !by_pdev.contains(info.pdev.as_str())
                {
                    continue;
                }
                let key = match info.client_id {
                    Some(id) => ClientKey::Id(info.pdev.clone(), id),
                    None => ClientKey::Fd(info.pdev.clone(), pid, name.clone()),
                };
                let at = *index.entry(key.clone()).or_insert_with(|| {
                    out.list.push((
                        key,
                        Client {
                            owners: BTreeSet::new(),
                            info,
                            render: true,
                            read_at,
                        },
                    ));
                    out.list.len() - 1
                });
                let client = &mut out.list[at].1;
                client.owners.insert(pid);
                client.render = client.render && target.starts_with("/dev/dri/renderD");
            }
            known.insert(pid, entry);
        }
        self.known = known;
        self.partial = out_of_time;
        if !out_of_time {
            self.cursor = 0;
        } else if let Some(c) = cursor {
            self.cursor = c; // else only new processes were deferred: the known ones keep their place
        }
        let mut notes = Vec::new();
        if out_of_time {
            notes.push(format!(
                "scan budget reached, {deferred} processes not rescanned this refresh (last known GPU fds kept)"
            ));
        }
        if denied > 0 {
            notes.push(format!(
                "{denied} processes not inspectable (other users' or kernel threads; sudo includes them)"
            ));
        }
        if !out.incomplete.is_empty() {
            notes.push(format!(
                "{} processes whose fds or fdinfo couldn't all be read",
                out.incomplete.len()
            ));
        }
        self.status = notes.join("; ");
        out
    }

    pub fn sample(
        &mut self,
    ) -> (
        BTreeMap<u64, BTreeMap<Pid, ProcGpu>>,
        BTreeMap<u64, AdapterGpu>,
    ) {
        let adapters = self.adapters();
        let clients = self.clients();
        let by_pdev: HashMap<&str, u64> = self
            .cards
            .iter()
            .map(|c| (c.pdev.as_str(), c.key))
            .collect();
        let unified: BTreeSet<&str> = self
            .cards
            .iter()
            .filter(|c| c.unified_video)
            .map(|c| c.pdev.as_str())
            .collect();
        let mut procs: BTreeMap<u64, BTreeMap<Pid, ProcGpu>> = self
            .cards
            .iter()
            .map(|c| (c.key, BTreeMap::new()))
            .collect();

        let mut by_pid: BTreeMap<(u64, Pid), Vec<usize>> = BTreeMap::new();
        for (i, (_, client)) in clients.list.iter().enumerate() {
            let Some(&key) = by_pdev.get(client.info.pdev.as_str()) else {
                continue;
            };
            for &pid in &client.owners {
                by_pid.entry((key, pid)).or_default().push(i);
            }
        }

        let now = self.now_ns();
        let mut current: HashMap<ClientKey, (BTreeMap<String, u64>, u128)> = HashMap::new();
        for (client_key, client) in &clients.list {
            let old = self.prev.get(client_key);
            let Some(engines) = &client.info.engines else {
                if let Some(old) = old {
                    current.insert(client_key.clone(), old.clone());
                }
                continue;
            };
            let mut high = old.map(|o| o.0.clone()).unwrap_or_default();
            for (name, &ns) in engines {
                // The DRM spec allows a counter to step back briefly.
                let h = high.entry(name.clone()).or_insert(0);
                *h = (*h).max(ns);
            }
            current.insert(client_key.clone(), (high, client.read_at));
        }

        for ((key, pid), held) in &by_pid {
            let (key, pid) = (*key, *pid);
            let held: Vec<&(ClientKey, Client)> = held.iter().map(|&i| &clients.list[i]).collect();
            // No commit counter in fdinfo, so Paged stays unavailable rather than 0. ROCm/HIP (a
            // /dev/kfd holder) submits compute through user-mode KFD queues the kernel doesn't time
            // per process, so its engine load is unknown, not 0 (rule 2); its VRAM is still in fdinfo.
            let kfd = clients.kfd.contains(&pid);
            let complete = !clients.incomplete.contains(&pid); // else any sum would look whole
            let sum = |get: fn(&FdInfo) -> Option<u64>| -> Option<u64> {
                if !complete {
                    return None;
                }
                held.iter()
                    .map(|(_, c)| get(&c.info))
                    .try_fold(0u64, |a, b| b.and_then(|b| a.checked_add(b)))
            };
            // Every buffer this process can reach, shared ones included: a reference, never summed.
            let resident_vram = sum(|i| i.vram);
            let resident_gtt = sum(|i| i.gtt);
            let missing = if complete {
                Reason::Legacy
            } else {
                Reason::Incomplete
            };
            let as_bytes = |v: Option<u64>| v.map_or(Bytes::unavailable(missing), Bytes::measured);

            // Bytes are charged to this process only when nothing else can hold them. Anything else
            // leaves the whole process's bytes in the adapter's unattributed part (rules 1 and 5).
            let infos = || held.iter().map(|(_, c)| &c.info);
            let reason = if !complete {
                Some(Reason::Incomplete)
            } else if held.iter().any(|(_, c)| c.owners.len() > 1) {
                Some(Reason::SharedClient)
            } else if infos().any(|i| i.shared.is_none()) {
                Some(Reason::Legacy)
            } else if infos().any(|i| i.shared.is_some_and(|s| s > 0)) {
                Some(Reason::SharedBuffers)
            } else if !held.iter().all(|(_, c)| c.render) {
                Some(Reason::Primary)
            } else if resident_vram.is_none() || resident_gtt.is_none() {
                Some(Reason::Legacy)
            } else if self.partial {
                Some(Reason::PartialScan)
            } else {
                None
            };
            let (vram, shared) = match reason {
                Some(r) => (Bytes::unavailable(r), Bytes::unavailable(r)),
                None => (as_bytes(resident_vram), as_bytes(resident_gtt)),
            };

            let mut engines: BTreeMap<String, f64> = BTreeMap::new();
            let mut unknown: Option<Reason> = None;
            for (client_key, client) in &held {
                let why = if kfd {
                    Some(Reason::KfdUntimed)
                } else if !complete {
                    Some(Reason::Incomplete)
                } else if self.partial {
                    Some(Reason::PartialScan)
                } else if client.owners.len() > 1 {
                    Some(Reason::SharedClient)
                } else if client.info.engines.is_none() {
                    Some(Reason::TornCounter)
                } else {
                    None
                };
                let old = match (why, self.prev.get(client_key)) {
                    (None, Some(old)) if client.read_at > old.1 => old,
                    (why, _) => {
                        unknown = Some(why.unwrap_or(Reason::NoBaseline));
                        break;
                    }
                };
                let delta = (client.read_at - old.1) as f64;
                let Some((high, _)) = current.get(client_key) else {
                    continue;
                };
                for (name, &ns) in high {
                    // amdgpu omits an engine until it has run, so a missing earlier value really is 0.
                    let before = old.0.get(name).copied().unwrap_or(0);
                    let spent = i128::from(ns) - i128::from(before);
                    let pct = (100.0 * spent as f64 / delta).clamp(0.0, 100.0);
                    let engine = engine_key(name, unified.contains(client.info.pdev.as_str()));
                    let total = engines.entry(engine).or_insert(0.0);
                    *total = (*total + pct).min(100.0);
                }
            }
            let engines = match unknown {
                Some(reason) => Engines::Unavailable { reason },
                None => Engines::Measured { by_engine: engines },
            };

            let row = ProcGpu {
                vram,
                shared,
                committed: Bytes::unavailable(Reason::NotOnBackend),
                engines,
                owner: None,
                resident: Some(Resident {
                    vram: as_bytes(resident_vram),
                    gtt: as_bytes(resident_gtt),
                }),
                kfd,
            };
            procs.entry(key).or_default().insert(pid, row);
        }

        // Keep baselines of clients a deferred scan missed, briefly.
        for (client_key, value) in &self.prev {
            if !current.contains_key(client_key) && now.saturating_sub(value.1) < PRUNE_NS {
                current.insert(client_key.clone(), value.clone());
            }
        }
        self.prev = current;
        (procs, adapters)
    }
}
