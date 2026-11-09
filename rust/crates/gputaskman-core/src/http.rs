//! The only network client: GET requests to loopback ports.
use crate::traits::{HttpGet, LocalUrl};
use std::time::Duration;

const MAX_BODY: u64 = 4 << 20;

pub struct UreqGet {
    agent: ureq::Agent,
}

impl Default for UreqGet {
    fn default() -> Self {
        let config = ureq::Agent::config_builder()
            .proxy(None)
            .http_status_as_error(false)
            .max_redirects(0)
            .build();
        Self {
            agent: config.into(),
        }
    }
}

impl HttpGet for UreqGet {
    fn get(&self, url: &LocalUrl, timeout: Duration) -> Option<String> {
        let mut response = self
            .agent
            .get(url.to_string())
            .header("Accept", "application/json")
            .config()
            .timeout_global(Some(timeout))
            .build()
            .call()
            .ok()?;
        if response.status().as_u16() != 200 {
            return None;
        }
        let bytes = response
            .body_mut()
            .with_config()
            .limit(MAX_BODY)
            .read_to_vec()
            .ok()?;
        Some(String::from_utf8_lossy(&bytes).into_owned())
    }
}
