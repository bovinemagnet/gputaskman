use gputaskman_core::provenance::{Bytes, MeasuredTotal};
fn f(b: Bytes) { let mut t = MeasuredTotal::new(); t.add(b); }
fn main() {}
