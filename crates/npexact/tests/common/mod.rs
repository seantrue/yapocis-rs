//! Shared helpers for the host-fixture tests: the host slug (which fixture set
//! this machine compares against) and `.npz` loading. Include with `mod common;`.
#![allow(dead_code)]

use ndarray::{ArrayD, IxDyn};
use ndarray_npy::NpzReader;
use std::fs::File;
use std::path::PathBuf;

/// `yapocis.codegen.host_fixtures.host_slug()`: `<system>-<machine>-<cpu brand>`,
/// lowercased, runs of other characters as `-`.
pub fn host_slug() -> String {
    let system = match std::env::consts::OS {
        "macos" => "Darwin".to_string(),
        "linux" => "Linux".to_string(),
        other => other.to_string(),
    };
    let machine = match std::env::consts::ARCH {
        "aarch64" if std::env::consts::OS == "macos" => "arm64".to_string(),
        other => other.to_string(),
    };
    slug(&format!("{system}-{machine}-{}", cpu_brand()))
}

fn cpu_brand() -> String {
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() {
                return s;
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in text.lines() {
                if line.to_lowercase().starts_with("model name") {
                    if let Some((_, v)) = line.split_once(':') {
                        return v.trim().to_string();
                    }
                }
            }
        }
    }
    std::env::consts::ARCH.to_string()
}

fn slug(text: &str) -> String {
    let mut out = String::new();
    for c in text.to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}

/// `fixtures/host/<host slug>` for this crate, if it was recorded on this host.
pub fn fixture_dir(crate_dir: &str) -> Option<PathBuf> {
    let dir = PathBuf::from(crate_dir)
        .join("fixtures/host")
        .join(host_slug());
    if dir.join("index.json").exists() {
        Some(dir)
    } else {
        eprintln!(
            "no host fixtures for {} ({}): record them with `yapocis-tools bundle --host-fixtures`",
            host_slug(),
            dir.display()
        );
        None
    }
}

pub fn open_npz(path: &std::path::Path) -> NpzReader<File> {
    NpzReader::new(File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap()
}

pub fn read_array<T: ndarray_npy::ReadableElement>(
    npz: &mut NpzReader<File>,
    name: &str,
) -> ArrayD<T> {
    npz.by_name::<ndarray::OwnedRepr<T>, IxDyn>(name)
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Exact comparison of two float slices by bit pattern (NaNs compare equal to
/// NaNs). Returns the first differing index.
pub fn first_bit_diff_f64(got: &[f64], want: &[f64]) -> Option<usize> {
    if got.len() != want.len() {
        return Some(got.len().min(want.len()));
    }
    got.iter()
        .zip(want)
        .position(|(a, b)| a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan()))
}

pub fn first_bit_diff_f32(got: &[f32], want: &[f32]) -> Option<usize> {
    if got.len() != want.len() {
        return Some(got.len().min(want.len()));
    }
    got.iter()
        .zip(want)
        .position(|(a, b)| a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan()))
}
