//! On-disk cache for compiled programs (OpenCL binaries, CUDA PTX).
//!
//! The directory is `$YAPOCIS_CACHE` when set, else the platform cache
//! directory plus `yapocis-rs`. Entries are keyed by a SHA-256 over the
//! rendered source, build options, backend, device identity and this crate's
//! version, so a driver upgrade or a template change never serves stale code.

use sha2::{Digest, Sha256};
use std::path::PathBuf;
use yapocis_bundle::Backend;

pub fn cache_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("YAPOCIS_CACHE") {
        if !dir.is_empty() {
            return Some(PathBuf::from(dir));
        }
    }
    dirs::cache_dir().map(|d| d.join("yapocis-rs"))
}

pub fn cache_key(backend: Backend, identity: &str, source: &str, options: &[String]) -> String {
    let mut h = Sha256::new();
    h.update(backend.name().as_bytes());
    h.update(b"\0");
    h.update(identity.as_bytes());
    h.update(b"\0");
    h.update(env!("CARGO_PKG_VERSION").as_bytes());
    h.update(b"\0");
    for o in options {
        h.update(o.as_bytes());
        h.update(b"\0");
    }
    h.update(source.as_bytes());
    format!("{:x}", h.finalize())
}

/// Path for a cached artifact, creating the backend directory. `None` when caching is off.
pub fn entry_path(backend: Backend, key: &str, ext: &str) -> Option<PathBuf> {
    let dir = cache_dir()?.join(backend.name());
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join(format!("{key}.{ext}")))
}

pub fn read(backend: Backend, key: &str, ext: &str) -> Option<Vec<u8>> {
    let path = entry_path(backend, key, ext)?;
    std::fs::read(path).ok()
}

pub fn write(backend: Backend, key: &str, ext: &str, data: &[u8]) {
    if let Some(path) = entry_path(backend, key, ext) {
        let tmp = path.with_extension(format!("{ext}.tmp{}", std::process::id()));
        if std::fs::write(&tmp, data).is_ok() {
            let _ = std::fs::rename(&tmp, &path);
        }
    }
}
