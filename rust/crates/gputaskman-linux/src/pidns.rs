//! PID namespaces. Inside a container NVML reports *host* PIDs, which mean nothing in the container's
//! `/proc`. Both namespaces number processes in creation order, so the container's CUDA processes are
//! matched to NVML's unresolved PIDs by order.
use gputaskman_core::provenance::Confidence;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// The kernel's fixed inode number for the initial (host) PID namespace.
pub const PROC_PID_INIT_INO: u64 = 0xEFFF_FFFC;

/// Matches host PIDs to container PIDs. A match is sure (consistent) when a new NVML PID and a new local
/// CUDA process appear together (kept while both live) or the counts are equal; otherwise newest is paired
/// with newest as an order guess, and leftover NVML PIDs belong to processes outside this container.
#[derive(Debug, Default)]
pub struct PidMatcher {
    pairs: BTreeMap<u32, u32>,
    prev_host: Option<BTreeSet<u32>>,
    prev_local: BTreeSet<u32>,
}

impl PidMatcher {
    /// Returns `({host pid: (local pid, confidence)}, host pids left outside)`.
    pub fn update(
        &mut self,
        host: &[u32],
        local: &[u32],
    ) -> (BTreeMap<u32, (u32, Confidence)>, Vec<u32>) {
        let hs: BTreeSet<u32> = host.iter().copied().collect();
        let ls: BTreeSet<u32> = local.iter().copied().collect();
        self.pairs.retain(|h, l| hs.contains(h) && ls.contains(l));
        if let Some(prev_host) = &self.prev_host {
            let paired_local: BTreeSet<u32> = self.pairs.values().copied().collect();
            let new_h: Vec<u32> = host
                .iter()
                .copied()
                .filter(|h| !prev_host.contains(h) && !self.pairs.contains_key(h))
                .collect();
            let new_l: Vec<u32> = local
                .iter()
                .copied()
                .filter(|p| !self.prev_local.contains(p) && !paired_local.contains(p))
                .collect();
            if let ([h], [l]) = (new_h.as_slice(), new_l.as_slice()) {
                self.pairs.insert(*h, *l);
            }
        }
        self.prev_host = Some(hs);
        self.prev_local = ls;
        let mut out: BTreeMap<u32, (u32, Confidence)> = self
            .pairs
            .iter()
            .map(|(h, l)| (*h, (*l, Confidence::Consistent)))
            .collect();
        let paired_local: BTreeSet<u32> = self.pairs.values().copied().collect();
        let mut rest_h: Vec<u32> = host
            .iter()
            .copied()
            .filter(|h| !self.pairs.contains_key(h))
            .collect();
        rest_h.sort_unstable_by(|a, b| b.cmp(a)); // newest first
        let mut rest_l: Vec<u32> = local
            .iter()
            .copied()
            .filter(|p| !paired_local.contains(p))
            .collect();
        rest_l.sort_unstable_by(|a, b| b.cmp(a));
        let confidence = if rest_h.len() == rest_l.len() {
            Confidence::Consistent
        } else {
            Confidence::Order
        };
        for (h, l) in rest_h.iter().zip(rest_l.iter()) {
            out.insert(*h, (*l, confidence));
        }
        let leftover = rest_h.split_off(rest_l.len().min(rest_h.len()));
        (out, leftover)
    }
}

/// What the NVML backend needs to know about this machine's `/proc`.
pub trait NamespaceProbe: Send {
    /// True inside a container with its own PID namespace, where NVML reports host PIDs. False in the
    /// host namespace. `None` when it cannot be told.
    fn in_child_pid_namespace(&self) -> Option<bool>;
    /// Local PIDs with an NVIDIA device file open.
    fn nvidia_holders(&self) -> BTreeSet<u32>;
    /// libcuda mapped: the process has (or is creating) a CUDA context.
    fn has_cuda(&self, pid: u32) -> bool;
    fn exists(&self, pid: u32) -> bool;
    fn self_pid(&self) -> u32;
}

#[derive(Debug, Clone)]
pub struct ProcNamespace {
    pub proc_root: PathBuf,
}

impl ProcNamespace {
    pub fn new(proc_root: impl AsRef<Path>) -> Self {
        Self {
            proc_root: proc_root.as_ref().to_path_buf(),
        }
    }
}

impl NamespaceProbe for ProcNamespace {
    fn in_child_pid_namespace(&self) -> Option<bool> {
        let link = std::fs::read_link(self.proc_root.join("self/ns/pid")).ok()?;
        let text = link.to_string_lossy();
        let inode: u64 = text.split_once('[')?.1.trim_end_matches(']').parse().ok()?;
        Some(inode != PROC_PID_INIT_INO)
    }

    fn nvidia_holders(&self) -> BTreeSet<u32> {
        let mut pids = BTreeSet::new();
        let Ok(entries) = std::fs::read_dir(&self.proc_root) else {
            return pids;
        };
        for entry in entries.flatten() {
            let Some(pid) = entry
                .file_name()
                .to_str()
                .and_then(|n| n.parse::<u32>().ok())
            else {
                continue;
            };
            let Ok(fds) = std::fs::read_dir(entry.path().join("fd")) else {
                continue; // another user's process, or it exited
            };
            let holds = fds.flatten().any(|fd| {
                std::fs::read_link(fd.path())
                    .is_ok_and(|t| t.to_string_lossy().starts_with("/dev/nvidia"))
            });
            if holds {
                pids.insert(pid);
            }
        }
        pids
    }

    fn has_cuda(&self, pid: u32) -> bool {
        std::fs::read(self.proc_root.join(pid.to_string()).join("maps"))
            .map(|b| String::from_utf8_lossy(&b).contains("/libcuda.so"))
            .unwrap_or(false)
    }

    fn exists(&self, pid: u32) -> bool {
        self.proc_root.join(pid.to_string()).exists()
    }

    fn self_pid(&self) -> u32 {
        std::process::id()
    }
}

/// A scripted namespace for tests.
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug, Clone)]
pub struct FakeNamespace {
    child: Option<bool>,
    holders: BTreeSet<u32>,
    cuda: BTreeSet<u32>,
    existing: BTreeSet<u32>,
    me: u32,
}

#[cfg(any(test, feature = "test-support"))]
impl FakeNamespace {
    fn base(child: Option<bool>, holders: BTreeSet<u32>, cuda: BTreeSet<u32>) -> Self {
        Self {
            child,
            holders,
            cuda,
            existing: BTreeSet::new(),
            me: u32::MAX,
        }
    }

    /// The host PID namespace: NVML's PIDs are this machine's PIDs.
    pub fn bare_metal(holders: impl IntoIterator<Item = u32>) -> Self {
        Self::base(Some(false), holders.into_iter().collect(), BTreeSet::new())
    }

    /// A container's own PID namespace: NVML reports host PIDs.
    pub fn child(
        holders: impl IntoIterator<Item = u32>,
        cuda: impl IntoIterator<Item = u32>,
    ) -> Self {
        Self::base(
            Some(true),
            holders.into_iter().collect(),
            cuda.into_iter().collect(),
        )
    }

    /// The namespace cannot be told; only PIDs in `existing` exist locally.
    pub fn unknown(existing: impl IntoIterator<Item = u32>) -> Self {
        let mut ns = Self::base(None, BTreeSet::new(), BTreeSet::new());
        ns.existing = existing.into_iter().collect();
        ns
    }

    pub fn with_cuda(mut self, cuda: impl IntoIterator<Item = u32>) -> Self {
        self.cuda = cuda.into_iter().collect();
        self
    }

    pub fn with_holders(mut self, holders: impl IntoIterator<Item = u32>) -> Self {
        self.holders = holders.into_iter().collect();
        self
    }
}

#[cfg(any(test, feature = "test-support"))]
impl NamespaceProbe for FakeNamespace {
    fn in_child_pid_namespace(&self) -> Option<bool> {
        self.child
    }
    fn nvidia_holders(&self) -> BTreeSet<u32> {
        self.holders.clone()
    }
    fn has_cuda(&self, pid: u32) -> bool {
        self.cuda.contains(&pid)
    }
    fn exists(&self, pid: u32) -> bool {
        self.existing.contains(&pid)
    }
    fn self_pid(&self) -> u32 {
        self.me
    }
}
