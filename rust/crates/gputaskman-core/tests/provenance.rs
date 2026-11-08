#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::provenance::*;

#[test]
fn only_measured_bytes_reach_the_total() {
    let rows = [
        Bytes::measured(3),
        Bytes::Inferred {
            bytes: 100,
            by: Confidence::Consistent,
        },
        Bytes::Estimated {
            bytes: 100,
            from: EstimateSource::Gguf,
        },
        Bytes::Reported {
            bytes: 100,
            by: Reporter::LoadLog,
        },
        Bytes::measured(0),
        Bytes::measured(4),
    ];
    let mut total = MeasuredTotal::new();
    for m in rows.iter().filter_map(Bytes::as_measured) {
        total.add(m);
    }
    assert_eq!(total.total(), 7);
}

#[test]
fn measured_zero_is_not_unknown() {
    let zero = Bytes::measured(0);
    assert_eq!(zero.value(), Some(0));
    assert!(zero.is_measured());
    let unknown = Bytes::unavailable(Reason::NvmlHidden);
    assert_eq!(unknown.value(), None);
    assert!(unknown.as_measured().is_none());
}

#[test]
fn serialises_with_status_tag() {
    let cases = [
        (Bytes::measured(5), r#"{"status":"measured","bytes":5}"#),
        (
            Bytes::Inferred {
                bytes: 5,
                by: Confidence::Order,
            },
            r#"{"status":"inferred","bytes":5,"by":"order"}"#,
        ),
        (
            Bytes::Estimated {
                bytes: 5,
                from: EstimateSource::Gguf,
            },
            r#"{"status":"estimated","bytes":5,"from":"gguf"}"#,
        ),
        (
            Bytes::Reported {
                bytes: 5,
                by: Reporter::LoadLog,
            },
            r#"{"status":"reported","bytes":5,"by":"load_log"}"#,
        ),
        (
            Bytes::unavailable(Reason::SharedClient),
            r#"{"status":"unavailable","reason":"shared_client"}"#,
        ),
    ];
    for (value, json) in cases {
        assert_eq!(serde_json::to_string(&value).unwrap(), json);
        assert_eq!(serde_json::from_str::<Bytes>(json).unwrap(), value);
    }
}

#[test]
fn reason_codes_match_python() {
    let codes = [
        (Reason::Legacy, "legacy"),
        (Reason::SharedClient, "shared_client"),
        (Reason::SharedBuffers, "shared_buffers"),
        (Reason::Primary, "primary"),
        (Reason::Incomplete, "incomplete"),
        (Reason::PartialScan, "partial_scan"),
        (Reason::NvmlHidden, "nvml_hidden"),
        (Reason::V1Unknown, "v1_unknown"),
    ];
    for (reason, code) in codes {
        assert_eq!(reason.code(), code);
    }
    assert!(!Reason::NvmlHidden.voids_attribution());
    assert!(Reason::SharedBuffers.voids_attribution());
}

#[test]
fn markers() {
    assert_eq!(Confidence::Consistent.marker(), "*");
    assert_eq!(Confidence::Order.marker(), "?");
}
