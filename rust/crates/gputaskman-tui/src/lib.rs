//! gputaskman's interactive terminal UI: a live view of what is inside each GPU.
pub mod action;
pub mod app;
#[cfg(feature = "test-support")]
pub mod fixtures;
pub mod format;
pub mod history;
pub mod rows;
pub mod sampler;
mod terminal;
pub mod view;
pub mod widgets;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::Duration;

use crossterm::event::{self, Event, KeyEventKind};
use gputaskman_core::collector::Snapshot;

use crate::app::App;
use crate::sampler::Sampler;
pub use crate::sampler::Source;

/// Before the first sample: NVML and DRM engine rates need a baseline (Python waits 0.5 s).
const WARMUP: Duration = Duration::from_millis(500);
/// How often input is checked while nothing arrives.
const POLL: Duration = Duration::from_millis(100);

pub struct Config {
    pub interval: Duration,
}

#[derive(Debug)]
pub enum TuiError {
    Io(std::io::Error),
    /// The sampling thread stopped; the message is its panic's.
    Sampler(String),
}

impl std::fmt::Display for TuiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TuiError::Io(e) => write!(f, "terminal error: {e}"),
            TuiError::Sampler(m) => write!(f, "sampling stopped: {m}"),
        }
    }
}

impl std::error::Error for TuiError {}

impl From<std::io::Error> for TuiError {
    fn from(e: std::io::Error) -> Self {
        TuiError::Io(e)
    }
}

/// Runs the TUI until the user quits. `sources` are one collector per GPU, in GPU order.
pub fn run(sources: Vec<Box<dyn Source>>, config: Config) -> Result<(), TuiError> {
    let paused = Arc::new(AtomicBool::new(false));
    let (sampler, rx) = Sampler::spawn(sources, config.interval, WARMUP, Arc::clone(&paused))?;
    let (guard, mut terminal) = terminal::TerminalGuard::enter()?;
    let result = event_loop(&mut terminal, &rx, &paused);
    drop(guard); // the terminal is normal again before anything is reported
    match result {
        Err(TuiError::Sampler(_)) => Err(TuiError::Sampler(match sampler.join() {
            Err(message) => message,
            Ok(()) => "the sampling thread stopped".to_string(),
        })),
        other => other, // dropping `sampler` stops and joins it
    }
}

fn event_loop(
    terminal: &mut terminal::Tui,
    rx: &Receiver<Vec<Snapshot>>,
    paused: &AtomicBool,
) -> Result<(), TuiError> {
    let mut app = App::default();
    let mut dirty = true;
    loop {
        loop {
            match rx.try_recv() {
                Ok(snaps) => {
                    app.apply_all(snaps);
                    dirty = true;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return Err(TuiError::Sampler(String::new())),
            }
        }
        if dirty {
            let mut areas = app.areas;
            terminal.draw(|f| areas = view::draw(&app, f))?;
            app.areas = areas;
            dirty = false;
        }
        if event::poll(POLL)? {
            let action = match event::read()? {
                Event::Key(k) if k.kind == KeyEventKind::Press => action::from_key(k, &app),
                Event::Mouse(m) => action::from_mouse(m, &app),
                Event::Resize(..) => {
                    dirty = true;
                    None
                }
                _ => None,
            };
            if let Some(a) = action {
                app.update(a);
                paused.store(app.paused, Ordering::Relaxed);
                dirty = true;
            }
            if app.quit {
                return Ok(());
            }
        }
    }
}
