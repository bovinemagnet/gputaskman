//! Raw mode, the alternate screen and mouse capture, undone on every exit path.
use std::io::{self, Stdout};
use std::thread::{self, ThreadId};

use crossterm::cursor::Show;
use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

pub type Tui = Terminal<CrosstermBackend<Stdout>>;

pub fn restore() {
    let _ = execute!(
        io::stdout(),
        DisableMouseCapture,
        LeaveAlternateScreen,
        Show
    );
    let _ = disable_raw_mode();
}

/// Restores the terminal before a main-thread panic prints. A sampler-thread panic is reported by
/// `run` after the terminal is restored, so its default message is suppressed.
fn install_panic_hook(main: ThreadId) {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if thread::current().id() == main {
            restore();
            previous(info);
        }
    }));
}

pub struct TerminalGuard;

impl TerminalGuard {
    pub fn enter() -> io::Result<(TerminalGuard, Tui)> {
        install_panic_hook(thread::current().id());
        enable_raw_mode()?;
        let guard = TerminalGuard; // from here on, Drop restores
        execute!(io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
        let terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
        Ok((guard, terminal))
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore();
    }
}
