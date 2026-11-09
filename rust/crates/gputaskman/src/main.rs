//! gputaskman: an htop for what's inside your GPU.

#[cfg(target_os = "linux")]
mod cli;

#[cfg(target_os = "linux")]
fn main() {
    std::process::exit(cli::run());
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("gputaskman's Rust build supports native Linux only in this milestone");
    std::process::exit(1);
}
