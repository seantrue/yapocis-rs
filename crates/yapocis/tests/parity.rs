//! Replay the Python-generated parity fixtures (`yapocis-tools bundle --parity`)
//! through the Rust runtime on the same backend and compare outputs.

use ndarray::{ArrayD, IxDyn};
use ndarray_npy::NpzReader;
use serde::Deserialize;
use std::fs::File;
use std::path::PathBuf;
use yapocis::rt::{CallArg, CallOptions, HostView, OutputValue};
use yapocis::{available_backends, Backend, DType, Device, MANIFEST};

mod common;

/// `bundle/parity/<backend>/<device>`: fixtures recorded on this device.
fn parity_dir(device: &Device) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("bundle/parity")
        .join(device.backend().name())
        .join(common::device_slug(&device.name()))
}

#[derive(Deserialize)]
struct Index {
    backend: String,
    rtol: f64,
    atol: f64,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    interface: String,
    kernel: String,
    file: String,
    args: Vec<ArgRef>,
    outputs: Vec<String>,
    #[serde(default)]
    global_size: Option<usize>,
}

#[derive(Deserialize)]
struct ArgRef {
    #[serde(default)]
    array: Option<String>,
    #[serde(default)]
    scalar: Option<f64>,
}

/// A loaded numpy array as raw bytes plus dtype and shape.
struct Npy {
    dtype: DType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn read_any(npz: &mut NpzReader<File>, name: &str) -> Npy {
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
    try_type!(i32, DType::Int32);
    try_type!(i16, DType::Int16);
    try_type!(u32, DType::Uint32);
    try_type!(f64, DType::Float64);
    try_type!(num_complex::Complex<f32>, DType::Complex64);
    panic!("{name}: unsupported dtype in npz");
}

fn allclose(dtype: DType, got: &[u8], want: &[u8], rtol: f64, atol: f64) -> Result<(), String> {
    macro_rules! cmp {
        ($t:ty, $to_f64:expr) => {{
            let g: &[$t] = bytemuck::cast_slice(got);
            let w: &[$t] = bytemuck::cast_slice(want);
            if g.len() != w.len() {
                return Err(format!("length {} vs {}", g.len(), w.len()));
            }
            let mut worst = 0f64;
            let mut worst_i = 0;
            for (i, (x, y)) in g.iter().zip(w).enumerate() {
                let (x, y): (f64, f64) = ($to_f64(*x), $to_f64(*y));
                if x.is_nan() && y.is_nan() {
                    continue;
                }
                let d = (x - y).abs();
                let tol = atol + rtol * y.abs();
                if d > tol && d - tol > worst {
                    worst = d - tol;
                    worst_i = i;
                }
            }
            if worst > 0.0 {
                let (x, y): (f64, f64) = ($to_f64(g[worst_i]), $to_f64(w[worst_i]));
                return Err(format!("element {worst_i}: got {x}, want {y}"));
            }
            Ok(())
        }};
    }
    match dtype {
        DType::Float32 => cmp!(f32, |v: f32| v as f64),
        DType::Float64 => cmp!(f64, |v: f64| v),
        DType::Int32 => cmp!(i32, |v: i32| v as f64),
        DType::Int16 => cmp!(i16, |v: i16| v as f64),
        DType::Uint32 => cmp!(u32, |v: u32| v as f64),
        DType::Complex64 => {
            let g: &[f32] = bytemuck::cast_slice(got);
            let w: &[f32] = bytemuck::cast_slice(want);
            let gb = bytemuck::cast_slice::<f32, u8>(g);
            let wb = bytemuck::cast_slice::<f32, u8>(w);
            allclose(DType::Float32, gb, wb, rtol, atol)
        }
        other => Err(format!("no comparison for {other}")),
    }
}

fn replay(backend: Backend) -> Option<(usize, Vec<String>)> {
    let device = match Device::open(backend, None) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping {backend:?}: {e}");
            return None;
        }
    };
    let dir = parity_dir(&device);
    let index_path = dir.join("index.json");
    if !index_path.exists() {
        eprintln!(
            "no parity fixtures for {backend:?} on {} at {}",
            device.name(),
            index_path.display()
        );
        return None;
    }
    let index: Index =
        serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
    assert_eq!(index.backend, backend.name());
    let mut programs = std::collections::HashMap::new();
    let mut failures = Vec::new();
    let mut passed = 0;
    for case in &index.cases {
        let program = programs.entry(case.interface.clone()).or_insert_with(|| {
            let entry = MANIFEST
                .interface(&case.interface)
                .expect("interface in manifest");
            let module_source = entry.sources.get(backend.name()).expect("source");
            let text = std::fs::read_to_string(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("bundle")
                    .join(&module_source.path),
            )
            .unwrap();
            device.load(entry, &text).unwrap()
        });
        let mut npz = NpzReader::new(File::open(dir.join(&case.file)).unwrap()).unwrap();
        let inputs: Vec<Option<Npy>> = case
            .args
            .iter()
            .map(|a| a.array.as_deref().map(|name| read_any(&mut npz, name)))
            .collect();
        let mut args: Vec<CallArg<'_>> = case
            .args
            .iter()
            .zip(&inputs)
            .map(|(a, npy)| match (npy, a.scalar) {
                (Some(n), _) => CallArg::In(HostView {
                    dtype: n.dtype,
                    shape: &n.shape,
                    bytes: &n.bytes,
                }),
                (None, Some(s)) => CallArg::Scalar(s),
                _ => panic!("bad arg"),
            })
            .collect();
        let opts = CallOptions {
            global: case.global_size.map(|g| vec![g]),
            ..Default::default()
        };
        let label = format!("{}.{}", case.interface, case.kernel);
        let out = match program.call(&case.kernel, &mut args, &opts) {
            Ok(o) => o,
            Err(e) => {
                failures.push(format!("{label}: call failed: {e}"));
                continue;
            }
        };
        let got: Vec<_> = out
            .outputs
            .into_iter()
            .filter_map(|o| match o.value {
                OutputValue::Host {
                    shape,
                    dtype,
                    bytes,
                } => Some((shape, dtype, bytes)),
                _ => None,
            })
            .collect();
        if got.len() != case.outputs.len() {
            failures.push(format!(
                "{label}: {} outputs, expected {}",
                got.len(),
                case.outputs.len()
            ));
            continue;
        }
        let mut ok = true;
        for ((shape, dtype, bytes), name) in got.iter().zip(&case.outputs) {
            let want = read_any(&mut npz, name);
            if *shape != want.shape || *dtype != want.dtype {
                failures.push(format!(
                    "{label}: {name} shape/dtype {shape:?}/{dtype} vs {:?}/{}",
                    want.shape, want.dtype
                ));
                ok = false;
                continue;
            }
            if let Err(e) = allclose(*dtype, bytes, &want.bytes, index.rtol, index.atol) {
                failures.push(format!("{label}: {name}: {e}"));
                ok = false;
            }
        }
        if ok {
            passed += 1;
        }
    }
    Some((passed, failures))
}

#[test]
fn parity_with_python_outputs() {
    let mut any = false;
    for backend in available_backends() {
        if let Some((passed, failures)) = replay(backend) {
            any = true;
            println!("{backend:?}: {passed} cases match");
            assert!(
                failures.is_empty(),
                "{backend:?}: {} mismatches:\n{}",
                failures.len(),
                failures.join("\n")
            );
            assert!(passed > 50, "{backend:?}: only {passed} cases");
        }
    }
    assert!(any, "no backend with parity fixtures was available");
}

#[test]
fn device_slug_matches_python() {
    use common::device_slug;
    assert_eq!(device_slug("Apple M4 Max"), "apple-m4-max");
    assert_eq!(
        device_slug("NVIDIA GeForce RTX 3080"),
        "nvidia-geforce-rtx-3080"
    );
    assert_eq!(
        device_slug("  pthread-Intel(R) Core(TM) i9  "),
        "pthread-intel-r-core-tm-i9"
    );
}
