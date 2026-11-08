//! Seams between the collector and the platform: clocks, HTTP, processes and GPU backends.
use crate::model::{Adapter, AdapterGpu, BackendKind, DeviceTelemetry, Pid, ProcGpu};
use crate::provenance::Reason;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub trait Clock: Send + Sync {
    /// Monotonic time since an arbitrary fixed origin.
    fn mono(&self) -> Duration;
    /// Wall-clock seconds since the Unix epoch.
    fn wall(&self) -> f64;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn mono(&self) -> Duration {
        static ORIGIN: OnceLock<Instant> = OnceLock::new();
        ORIGIN.get_or_init(Instant::now).elapsed()
    }

    fn wall(&self) -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanPolicy {
    Budgeted,
    Unbounded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Deadline {
    pub at: Option<Duration>,
}

impl Deadline {
    pub fn after(clock: &dyn Clock, budget: Duration) -> Self {
        Self {
            at: Some(clock.mono() + budget),
        }
    }

    pub fn none() -> Self {
        Self { at: None }
    }

    pub fn passed(&self, clock: &dyn Clock) -> bool {
        self.at.is_some_and(|at| clock.mono() >= at)
    }
}

/// A URL on this machine's loopback interface. There is no way to name another host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalUrl {
    port: u16,
    path: String,
}

impl LocalUrl {
    pub fn loopback(port: u16, path: &str) -> Self {
        // Anchor the path so it can never turn into userinfo and change the host.
        let path = if path.starts_with('/') {
            path.to_string()
        } else {
            format!("/{path}")
        };
        Self { port, path }
    }

    pub fn base(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
}

impl fmt::Display for LocalUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.base(), self.path)
    }
}

pub trait HttpGet: Send + Sync {
    /// `None` unless the status is 200. The body is capped at 4 MiB.
    fn get(&self, url: &LocalUrl, timeout: Duration) -> Option<String>;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StartTime {
    pub ticks: u64,
    pub wall: f64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProcBasics {
    pub name: Option<String>,
    pub exe: Option<String>,
    pub cmdline: Option<Vec<String>>,
    pub start: Option<StartTime>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapEntry {
    pub path: String,
    pub rss: u64,
}

pub trait ProcessSource: Send + Sync {
    fn pids(&self) -> Vec<Pid>;
    /// `None` when the process cannot be opened at all.
    fn basics(&self, pid: Pid) -> Option<ProcBasics>;
    fn start_time(&self, pid: Pid) -> Option<StartTime>;
    fn parent(&self, pid: Pid) -> Option<Pid>;
    fn memory_maps(&self, pid: Pid) -> Result<Vec<MapEntry>, Reason>;
    fn listening_ports(&self) -> BTreeMap<Pid, Vec<u16>>;
    fn self_pid(&self) -> Pid;
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BackendSample {
    pub procs: BTreeMap<Pid, ProcGpu>,
    pub outside: BTreeMap<Pid, ProcGpu>,
    pub adapter: AdapterGpu,
    pub status: Option<String>,
}

pub trait GpuBackend: Send {
    fn kind(&self) -> BackendKind;
    fn adapters(&self) -> Vec<Adapter>;
    fn start(&mut self, adapter: &Adapter, policy: ScanPolicy);
    fn sample(&mut self) -> BackendSample;
    fn device(&mut self, index: usize) -> Result<DeviceTelemetry, Reason>;
    fn driver(&self) -> Option<String>;
    fn cuda(&self) -> Option<String>;
}
