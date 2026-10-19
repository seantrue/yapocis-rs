//! Shared helpers for the integration tests: available devices, bundle paths,
//! `.npz` loading and tolerant comparison. Include with `mod common;`.
#![allow(dead_code)]

use ndarray::{ArrayD, IxDyn};
use ndarray_npy::NpzReader;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use yapocis::{available_backends, Backend, DType, Device};

pub fn bundle_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bundle")
}

/// Directory name for a device, as `yapocis.codegen.parity.device_slug` makes
/// it: `"Apple M4 Max"` -> `"apple-m4-max"`.
pub fn device_slug(name: &str) -> String {
    let mut slug = String::new();
    for c in name.to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            slug.push(c);
        } else if !slug.ends_with('-') {
            slug.push('-');
        }
    }
    slug.trim_matches('-').to_string()
}

/// Slug of the device `backend` opens by default, opened once per process.
/// Fixtures are recorded per device, so this picks the set to compare against.
fn default_device_slug(backend: Backend) -> String {
    static SLUGS: OnceLock<Mutex<HashMap<&'static str, String>>> = OnceLock::new();
    let mut slugs = SLUGS.get_or_init(Default::default).lock().unwrap();
    slugs
        .entry(backend.name())
        .or_insert_with(|| match Device::open(backend, None) {
            Ok(d) => device_slug(&d.name()),
            Err(_) => "no-device".to_string(),
        })
        .clone()
}

/// `bundle/parity/lib/<backend>/<device>` for library-level fixtures recorded
/// on the device `backend` opens here.
pub fn parity_lib_dir(backend: Backend) -> PathBuf {
    bundle_dir()
        .join("parity/lib")
        .join(backend.name())
        .join(default_device_slug(backend))
}

/// Every backend compiled in that opens a device on this machine.
pub fn devices() -> Vec<Device> {
    available_backends()
        .into_iter()
        .filter_map(|b| match Device::open(b, None) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("skipping {b:?}: {e}");
                None
            }
        })
        .collect()
}

/// Parsed `index.json` of the library fixtures for this device, if any were
/// recorded on it; otherwise `None`, and the caller skips.
pub fn lib_index(backend: Backend) -> Option<serde_json::Value> {
    let path = parity_lib_dir(backend).join("index.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "no library parity fixtures for {backend:?} on this device ({})",
            path.display()
        );
        return None;
    };
    serde_json::from_str(&text).ok()
}

pub struct Npy {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

impl Npy {
    pub fn as_slice<T: bytemuck::Pod>(&self) -> &[T] {
        bytemuck::cast_slice(&self.bytes)
    }

    pub fn to_array<T: bytemuck::Pod>(&self) -> ArrayD<T> {
        ArrayD::from_shape_vec(IxDyn(&self.shape), self.as_slice::<T>().to_vec()).unwrap()
    }
}

pub fn open_npz(path: &std::path::Path) -> NpzReader<File> {
    NpzReader::new(File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap()
}

/// Load one array of an `.npz` whatever its dtype.
pub fn read_any(npz: &mut NpzReader<File>, name: &str) -> Npy {
    macro_rules! try_type {
        ($t:ty, $d:expr) => {
            if let Ok(a) = npz.by_name::<ndarray::OwnedRepr<$t>, IxDyn>(name) {
                let a: ArrayD<$t> = a;
                let v = a.as_standard_layout().to_owned();
                return Npy {
                    dtype: $d,
                    shape: v.shape().to_vec(),
                    bytes: bytemuck::cast_slice(v.as_slice().unwrap()).to_vec(),
                };
            }
        };
    }
    try_type!(f32, DType::Float32);
    try_type!(f64, DType::Float64);
    try_type!(i32, DType::Int32);
    try_type!(i16, DType::Int16);
    try_type!(i64, DType::Int64);
    try_type!(u8, DType::Uint8);
    try_type!(u32, DType::Uint32);
    try_type!(num_complex::Complex<f32>, DType::Complex64);
    panic!("{name}: unsupported dtype in npz");
}

/// Load a typed array from an `.npz`.
pub fn read_array<T: ndarray_npy::ReadableElement>(
    npz: &mut NpzReader<File>,
    name: &str,
) -> ArrayD<T> {
    npz.by_name::<ndarray::OwnedRepr<T>, IxDyn>(name)
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// numpy-style `allclose` over f32 arrays (NaNs equal); returns the worst offender.
pub fn allclose_f32(got: &[f32], want: &[f32], rtol: f64, atol: f64) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!("length {} vs {}", got.len(), want.len()));
    }
    let mut worst = 0f64;
    let mut worst_i = None;
    for (i, (&x, &y)) in got.iter().zip(want).enumerate() {
        if x.is_nan() && y.is_nan() {
            continue;
        }
        let (x, y) = (x as f64, y as f64);
        let d = (x - y).abs();
        let tol = atol + rtol * y.abs();
        if d > tol && d - tol > worst {
            worst = d - tol;
            worst_i = Some(i);
        }
    }
    match worst_i {
        None => Ok(()),
        Some(i) => Err(format!("element {i}: got {}, want {}", got[i], want[i])),
    }
}

/// Fraction of elements that differ by more than `tol` (for rasterised images
/// where a few edge pixels may legitimately differ).
pub fn mismatch_fraction(got: &[f32], want: &[f32], tol: f32) -> f64 {
    assert_eq!(got.len(), want.len());
    let bad = got
        .iter()
        .zip(want)
        .filter(|(a, b)| (**a - **b).abs() > tol)
        .count();
    bad as f64 / got.len().max(1) as f64
}
