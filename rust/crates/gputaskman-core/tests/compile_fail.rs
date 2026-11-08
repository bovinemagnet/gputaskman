#![allow(clippy::unwrap_used, clippy::expect_used)]
#[test]
fn provenance_misuse_does_not_compile() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
