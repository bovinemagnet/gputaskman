//! Fakes for tests in this and later crates (feature `test-support`).
use crate::model::{Adapter, BackendKind, DeviceTelemetry, Pid};
use crate::provenance::Reason;
use crate::traits::*;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A manual clock. With `auto_advance`, every `mono()` call returns the current
/// value and then moves forward by the step.
pub struct FakeClock {
    state: Mutex<(Duration, f64)>,
    step: Mutex<Duration>,
}

impl FakeClock {
    pub fn new(wall: f64) -> Self {
        Self {
            state: Mutex::new((Duration::ZERO, wall)),
            step: Mutex::new(Duration::ZERO),
        }
    }

    pub fn advance(&self, d: Duration) {
        let mut s = lock(&self.state);
        s.0 += d;
        s.1 += d.as_secs_f64();
    }

    pub fn auto_advance(self, d: Duration) -> Self {
        *lock(&self.step) = d;
        self
    }
}

impl Clock for FakeClock {
    fn mono(&self) -> Duration {
        let step = *lock(&self.step);
        let mut s = lock(&self.state);
        let now = s.0;
        s.0 += step;
        s.1 += step.as_secs_f64();
        now
    }

    fn wall(&self) -> f64 {
        lock(&self.state).1
    }
}

#[derive(Default)]
pub struct FakeHttp {
    routes: BTreeMap<String, (u16, String)>,
    log: Mutex<Vec<String>>,
}

impl FakeHttp {
    pub fn route(mut self, port: u16, path: &str, status: u16, body: &str) -> Self {
        let key = LocalUrl::loopback(port, path).to_string();
        self.routes.insert(key, (status, body.to_string()));
        self
    }

    pub fn requests(&self) -> Vec<String> {
        lock(&self.log).clone()
    }
}

impl HttpGet for FakeHttp {
    fn get(&self, url: &LocalUrl, _timeout: Duration) -> Option<String> {
        let key = url.to_string();
        lock(&self.log).push(key.clone());
        match self.routes.get(&key) {
            Some((200, body)) => Some(body.clone()),
            _ => None,
        }
    }
}

#[derive(Default)]
pub struct FakeProcs {
    basics: BTreeMap<Pid, ProcBasics>,
    ports: BTreeMap<Pid, Vec<u16>>,
    maps: BTreeMap<Pid, Vec<MapEntry>>,
    parents: BTreeMap<Pid, Pid>,
}

impl FakeProcs {
    pub fn with(mut self, pid: Pid, basics: ProcBasics) -> Self {
        self.basics.insert(pid, basics);
        self
    }

    pub fn named(self, pid: Pid, name: &str) -> Self {
        self.with(
            pid,
            ProcBasics {
                name: Some(name.to_string()),
                exe: Some(format!("/opt/{name}")),
                cmdline: Some(vec![name.to_string()]),
                start: Some(StartTime {
                    ticks: 1,
                    wall: 1.0,
                }),
            },
        )
    }

    pub fn ports(mut self, pid: Pid, ports: &[u16]) -> Self {
        self.ports.insert(pid, ports.to_vec());
        self
    }

    pub fn maps(mut self, pid: Pid, maps: Vec<MapEntry>) -> Self {
        self.maps.insert(pid, maps);
        self
    }

    pub fn parent(mut self, pid: Pid, ppid: Pid) -> Self {
        self.parents.insert(pid, ppid);
        self
    }
}

impl ProcessSource for FakeProcs {
    fn pids(&self) -> Vec<Pid> {
        self.basics.keys().copied().collect()
    }

    fn basics(&self, pid: Pid) -> Option<ProcBasics> {
        self.basics.get(&pid).cloned()
    }

    fn start_time(&self, pid: Pid) -> Option<StartTime> {
        self.basics.get(&pid).and_then(|b| b.start)
    }

    fn parent(&self, pid: Pid) -> Option<Pid> {
        self.parents.get(&pid).copied()
    }

    fn memory_maps(&self, pid: Pid) -> Result<Vec<MapEntry>, Reason> {
        match self.maps.get(&pid) {
            Some(m) => Ok(m.clone()),
            None if self.basics.contains_key(&pid) => Ok(Vec::new()),
            None => Err(Reason::ProcessUnreadable),
        }
    }

    fn listening_ports(&self) -> BTreeMap<Pid, Vec<u16>> {
        self.ports.clone()
    }

    fn self_pid(&self) -> Pid {
        std::process::id()
    }
}

pub struct FakeBackend {
    kind: BackendKind,
    adapters: Vec<Adapter>,
    samples: Vec<BackendSample>,
    next: usize,
    starts: Arc<Mutex<usize>>,
    driver: Option<String>,
    cuda: Option<String>,
    device: Option<DeviceTelemetry>,
}

impl FakeBackend {
    pub fn new(kind: BackendKind, adapters: Vec<Adapter>) -> Self {
        Self {
            kind,
            adapters,
            samples: Vec::new(),
            next: 0,
            starts: Arc::new(Mutex::new(0)),
            driver: None,
            cuda: None,
            device: None,
        }
    }

    /// Replays the samples in order; the last one repeats.
    pub fn script(mut self, samples: Vec<BackendSample>) -> Self {
        self.samples = samples;
        self
    }

    /// Shared count of `start` calls.
    pub fn starts(&self) -> Arc<Mutex<usize>> {
        Arc::clone(&self.starts)
    }

    pub fn with_driver(mut self, driver: Option<String>, cuda: Option<String>) -> Self {
        self.driver = driver;
        self.cuda = cuda;
        self
    }

    pub fn with_device(mut self, device: DeviceTelemetry) -> Self {
        self.device = Some(device);
        self
    }
}

impl GpuBackend for FakeBackend {
    fn kind(&self) -> BackendKind {
        self.kind
    }

    fn adapters(&self) -> Vec<Adapter> {
        self.adapters.clone()
    }

    fn start(&mut self, _adapter: &Adapter, _policy: ScanPolicy) {
        *lock(&self.starts) += 1;
    }

    fn sample(&mut self) -> BackendSample {
        let Some(last) = self.samples.len().checked_sub(1) else {
            return BackendSample::default();
        };
        let sample = self.samples[self.next.min(last)].clone();
        self.next = (self.next + 1).min(last);
        sample
    }

    fn device(&mut self, _index: usize) -> Result<DeviceTelemetry, Reason> {
        self.device.clone().ok_or(Reason::NvmlNotLoaded)
    }

    fn driver(&self) -> Option<String> {
        self.driver.clone()
    }

    fn cuda(&self) -> Option<String> {
        self.cuda.clone()
    }
}
