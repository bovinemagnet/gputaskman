//! GPU runtime detection from loaded library names (`procinfo.py` `RUNTIMES`).
use std::collections::BTreeSet;
use std::sync::LazyLock;

use glob::Pattern;

/// (tag, library patterns: Windows DLLs, Linux shared objects with the version suffix removed, and
/// macOS dylibs, see [`lib_name`]). Order is display priority, most specific first.
pub const RUNTIMES: &[(&str, &[&str])] = &[
    (
        "llama.cpp",
        &[
            "llama.dll",
            "libllama*.dll",
            "llama-*.dll",
            "llm_engine*",
            "libllama*.so",
            "libllama*.dylib",
        ],
    ),
    (
        "whisper.cpp",
        &["whisper*.dll", "libwhisper*.so", "libwhisper*.dylib"],
    ),
    ("ggml", &["ggml*.dll", "libggml*.so", "libggml*.dylib"]),
    (
        "PyTorch",
        &[
            "torch_cuda*.dll",
            "torch_cpu.dll",
            "libtorch_cuda*.so",
            "libtorch_cpu.so",
            "libtorch_cpu.dylib",
        ],
    ),
    ("MLX", &["libmlx*.dylib"]),
    ("TensorRT", &["nvinfer*.dll", "libnvinfer*.so"]),
    (
        "ONNX RT",
        &[
            "onnxruntime*.dll",
            "libonnxruntime*.so",
            "libonnxruntime*.dylib",
        ],
    ),
    ("DirectML", &["directml.dll"]),
    ("HIP", &["amdhip64.dll", "amdhip64_*.dll", "libamdhip64.so"]),
    ("cuDNN", &["cudnn*.dll", "libcudnn*.so"]),
    (
        "CUDA",
        &[
            "nvcuda.dll",
            "nvcuda64.dll",
            "cudart64_*.dll",
            "cublas64_*.dll",
            "libcuda.so",
            "libcudart.so",
            "libcublas.so",
        ],
    ),
    // The Apple GPU's Metal driver bundle, mapped by every Metal process.
    ("Metal", &["agxmetal*"]),
    (
        "Vulkan",
        &["vulkan-1.dll", "libvulkan.so", "libmoltenvk.dylib"],
    ),
    ("D3D12", &["d3d12core.dll", "libd3d12core.so"]),
    (
        "OpenGL",
        &["opengl32.dll", "libgl.so", "libglx*.so", "libegl*.so"],
    ),
    ("D3D11", &["d3d11.dll"]),
    ("NVENC", &["nvencodeapi64.dll", "libnvidia-encode.so"]),
    ("NVDEC", &["nvcuvid.dll", "nvcuvid64.dll", "libnvcuvid.so"]),
];

type Compiled = Vec<(&'static str, Vec<Pattern>)>;

static COMPILED: LazyLock<Compiled> = LazyLock::new(|| {
    RUNTIMES
        .iter()
        .map(|(tag, pats)| {
            (
                *tag,
                pats.iter().filter_map(|p| Pattern::new(p).ok()).collect(),
            )
        })
        .collect()
});

/// Extension as `os.path.splitext` sees it: from the last dot, ignoring leading dots.
fn splitext(base: &str) -> &str {
    match base.rfind('.') {
        Some(i) if base[..i].chars().any(|c| c != '.') => &base[i..],
        _ => "",
    }
}

/// `^(.+\.so)(?:\.[0-9][0-9a-z.]*)?$`: the part up to `.so`, when the rest is a version suffix.
fn strip_so_version(base: &str) -> Option<&str> {
    let mut end = base.len();
    while let Some(i) = base[..end].rfind(".so") {
        let stop = i + 3;
        let rest = &base[stop..];
        let suffix_ok = rest.is_empty()
            || rest.strip_prefix('.').is_some_and(|s| {
                s.starts_with(|c: char| c.is_ascii_digit())
                    && s.chars()
                        .all(|c| c.is_ascii_digit() || c.is_ascii_lowercase() || c == '.')
            });
        if i >= 1 && suffix_ok {
            return Some(&base[..stop]);
        }
        if i == 0 {
            break;
        }
        end = stop - 1;
    }
    None
}

/// Library file name with any Linux version suffix removed (`libcudart.so.12.4` becomes
/// `libcudart.so`), or `None` when this is not a library. The caller lower-cases the name.
#[must_use]
pub fn lib_name(base: &str) -> Option<String> {
    if matches!(splitext(base), ".dll" | ".node" | ".dylib") || base.starts_with("agxmetal") {
        return Some(base.to_string());
    }
    strip_so_version(base).map(str::to_string)
}

/// Runtime tags whose patterns match any library name, in `RUNTIMES` order.
#[must_use]
pub fn match_runtimes(libs: &BTreeSet<String>) -> Vec<String> {
    COMPILED
        .iter()
        .filter(|(_, pats)| libs.iter().any(|l| pats.iter().any(|p| p.matches(l))))
        .map(|(tag, _)| (*tag).to_string())
        .collect()
}
