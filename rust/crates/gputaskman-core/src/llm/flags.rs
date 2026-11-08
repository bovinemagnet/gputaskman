//! Runtime flags read from a llama.cpp-style command line.
use serde::Serialize;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Flags {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ctx: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ngl: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub port: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ctk: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ctv: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_file: Option<String>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub no_mmap: bool,
}

fn flag_name(key: &str) -> Option<&'static str> {
    Some(match key {
        "-c" | "--ctx-size" => "ctx",
        "-ngl" | "--n-gpu-layers" | "--gpu-layers" => "ngl",
        "-np" | "--parallel" => "parallel",
        "-ctk" | "--cache-type-k" => "ctk",
        "-ctv" | "--cache-type-v" => "ctv",
        "--port" => "port",
        "--log-file" => "log_file",
        _ => return None,
    })
}

pub fn parse_flags(cmd: &[String]) -> Flags {
    let mut out = Flags::default();
    for (i, arg) in cmd.iter().enumerate() {
        let (mut key, mut val) = (arg.as_str(), None);
        if arg.starts_with("--")
            && let Some((k, v)) = arg.split_once('=')
        {
            key = k;
            val = Some(v);
        }
        if key == "--no-mmap" {
            out.no_mmap = true;
        } else if let Some(name) = flag_name(key) {
            if val.is_none() {
                val = cmd.get(i + 1).map(String::as_str);
            }
            let Some(val) = val else { continue };
            if matches!(name, "ctx" | "ngl" | "parallel" | "port") {
                let n = match val.trim().parse::<i64>() {
                    Ok(n) => Some(n),
                    Err(_) if name == "ngl" && (val == "all" || val == "auto") => Some(-1),
                    Err(_) => None,
                };
                if let Some(n) = n {
                    match name {
                        "ctx" => out.ctx = Some(n),
                        "ngl" => out.ngl = Some(n),
                        "parallel" => out.parallel = Some(n),
                        _ => out.port = Some(n),
                    }
                }
            } else {
                let v = Some(val.to_string());
                match name {
                    "ctk" => out.ctk = v,
                    "ctv" => out.ctv = v,
                    _ => out.log_file = v,
                }
            }
        }
    }
    out
}
