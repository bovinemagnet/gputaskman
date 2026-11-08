use gputaskman_core::provenance::{MeasuredTotal, Shown};
fn f(s: Shown) { let mut t = MeasuredTotal::new(); t.add(s); }
fn main() {}
