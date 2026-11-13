//! On-disk cache for compiled programs (OpenCL binaries, CUDA PTX).
//!
//! The directory is `$YAPOCIS_CACHE` when set, else the platform cache
//! directory plus `yapocis-rs`. Entries are keyed by a digest, in commonkit's
//! current algorithm (`HASH`/`COMMONKIT_HASH`: MD5 or BLAKE3), over the
//! rendered source, build options, backend, device identity and this crate's
//! version, so a driver upgrade or a template change never serves stale code.
//! A switch of algorithm makes old entries misses, which recompile.

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
    let mut h = commonkit::digest::Hasher::new();
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
    h.finish()
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

#[cfg(test)]
mod tests {
    use super::*;
    use commonkit::digest::{current, Algo};

    /// The program-cache key follows the commonkit switch: the known MD5 or BLAKE3 of a fixed
    /// program's key bytes, `ocl\0dev\00.1.0\0-O\0__kernel void k() {}` (at crate version
    /// 0.1.0; recompute both with `hashlib.md5` and `blake3.blake3` if the version moves).
    #[test]
    fn cache_key_is_the_current_algorithm() {
        assert_eq!(
            env!("CARGO_PKG_VERSION"),
            "0.1.0",
            "recompute the known answers"
        );
        let key = cache_key(
            Backend::Ocl,
            "dev",
            "__kernel void k() {}",
            &["-O".to_string()],
        );
        let want = match current() {
            Algo::Md5 => "2129f02defb99f785064f21824210b9b",
            _ => "38d8b1cb9623c3daf7a30b339ff797180fb0ce9ad63edd10493d6a64b931a968",
        };
        assert_eq!(key, want);
    }

    /// The suite runs isolated, and in the algorithm it was asked for (plan §3.1).
    #[test]
    fn hash_is_what_was_asked() {
        commonkit::testing::assert_isolated();
        if let Ok(asked) = std::env::var("COMMONKIT_HASH") {
            assert_eq!(
                Algo::from_name(asked.trim()),
                Some(current()),
                "COMMONKIT_HASH={asked}"
            );
        }
    }
}
