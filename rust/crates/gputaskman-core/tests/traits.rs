#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::http::UreqGet;
use gputaskman_core::testing::{FakeClock, FakeHttp};
use gputaskman_core::traits::*;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::Duration;

#[test]
fn local_url_is_loopback_only() {
    let url = LocalUrl::loopback(8080, "/props");
    assert_eq!(url.to_string(), "http://127.0.0.1:8080/props");
    assert_eq!(url.base(), "http://127.0.0.1:8080");
}

#[test]
fn deadline_follows_the_clock() {
    let clock = FakeClock::new(1000.0);
    let d = Deadline::after(&clock, Duration::from_millis(150));
    assert!(!d.passed(&clock));
    clock.advance(Duration::from_millis(150));
    assert!(d.passed(&clock));
    assert!(!Deadline::none().passed(&clock));
}

#[test]
fn fake_http_records_requests() {
    let http = FakeHttp::default().route(8080, "/props", 200, "{}");
    assert_eq!(
        http.get(
            &LocalUrl::loopback(8080, "/props"),
            Duration::from_millis(500)
        )
        .as_deref(),
        Some("{}")
    );
    assert_eq!(
        http.get(
            &LocalUrl::loopback(8080, "/nope"),
            Duration::from_millis(500)
        ),
        None
    );
    assert_eq!(
        http.requests(),
        vec!["http://127.0.0.1:8080/props", "http://127.0.0.1:8080/nope"]
    );
}

fn serve_once(status: &'static str, body: &'static str) -> (u16, std::thread::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0u8; 2048];
        let n = stream.read(&mut buf).unwrap();
        let request = String::from_utf8_lossy(&buf[..n]).to_string();
        let response = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(response.as_bytes()).unwrap();
        request
    });
    (port, handle)
}

#[test]
fn ureq_get_is_a_get_and_only_accepts_200() {
    let (port, server) = serve_once("200 OK", r#"{"ok":true}"#);
    let body = UreqGet::default().get(
        &LocalUrl::loopback(port, "/props"),
        Duration::from_millis(500),
    );
    assert_eq!(body.as_deref(), Some(r#"{"ok":true}"#));
    let request = server.join().unwrap();
    assert!(request.starts_with("GET /props HTTP/1.1"), "{request}");

    let (port, server) = serve_once("404 Not Found", "missing");
    assert_eq!(
        UreqGet::default().get(&LocalUrl::loopback(port, "/x"), Duration::from_millis(500)),
        None
    );
    server.join().unwrap();
}

#[test]
fn loopback_path_cannot_change_the_host() {
    assert_eq!(
        LocalUrl::loopback(80, "@evil.com/").to_string(),
        "http://127.0.0.1:80/@evil.com/"
    );
    assert_eq!(
        LocalUrl::loopback(80, "/ok").to_string(),
        "http://127.0.0.1:80/ok"
    );
}

#[test]
fn ureq_get_does_not_follow_redirects() {
    // The redirect target is a second loopback listener, so a followed redirect is observable.
    let target = TcpListener::bind("127.0.0.1:0").unwrap();
    target.set_nonblocking(true).unwrap();
    let target_port = target.local_addr().unwrap().port();
    let first = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = first.local_addr().unwrap().port();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = first.accept().unwrap();
        let mut buf = [0u8; 2048];
        let _ = stream.read(&mut buf).unwrap();
        let response = format!(
            "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:{target_port}/\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
        stream.write_all(response.as_bytes()).unwrap();
    });
    assert_eq!(
        UreqGet::default().get(&LocalUrl::loopback(port, "/r"), Duration::from_millis(500)),
        None
    );
    server.join().unwrap();
    std::thread::sleep(Duration::from_millis(100));
    assert!(
        target.accept().is_err(),
        "the redirect target was contacted"
    );
}
