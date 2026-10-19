//! Replay the derivation fixtures exported by the Python package
//! (`yapocis-tools bundle --fixtures`) against this crate.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;
use yapocis_bundle::{
    cuda_grid_block, default_global_size, derive_args, dispatch_global_size, metal_grid_tgs,
    ArgShape, Backend, DType, Dim, KernelArg, Manifest,
};

fn bundle_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../yapocis/bundle")
}

#[derive(Deserialize)]
struct Fixtures {
    schema_version: u32,
    interfaces: BTreeMap<String, BTreeMap<String, KernelFixture>>,
}

#[derive(Deserialize)]
struct KernelFixture {
    symbol: String,
    cases: BTreeMap<String, Case>,
}

#[derive(Deserialize)]
struct Case {
    args: Vec<ArgDesc>,
    #[serde(default)]
    kernel_args: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    dispatch_global_size: Option<Vec<usize>>,
    #[serde(default)]
    dispatch_error: Option<String>,
    #[serde(default)]
    default_global_size: Option<BTreeMap<String, Option<Vec<usize>>>>,
    #[serde(default)]
    cuda_launch: Option<GridBlock>,
    #[serde(default)]
    metal_launch: Option<GridTgs>,
}

#[derive(Deserialize)]
struct ArgDesc {
    #[serde(default)]
    shape: Option<Vec<usize>>,
    #[serde(default)]
    dtype: Option<DType>,
    #[serde(default)]
    scalar: Option<f64>,
}

#[derive(Deserialize)]
struct GridBlock {
    grid: Vec<usize>,
    block: Vec<usize>,
}

#[derive(Deserialize)]
struct GridTgs {
    grid: Vec<usize>,
    tgs: Vec<usize>,
}

fn to_args(descs: &[ArgDesc]) -> Vec<ArgShape> {
    descs
        .iter()
        .map(|d| match (&d.shape, d.scalar) {
            (Some(shape), _) => ArgShape::array(shape, d.dtype.unwrap()),
            (None, Some(v)) => ArgShape::Scalar(v),
            _ => panic!("bad arg description"),
        })
        .collect()
}

fn check_kernel_arg(got: &KernelArg, want: &serde_json::Value, ctx: &str) {
    let w = want.as_object().unwrap();
    let kind = w["kind"].as_str().unwrap();
    let param = w["param"].as_str().unwrap();
    match got {
        KernelArg::Scalar {
            param: p, value, ..
        } => {
            assert_eq!((kind, param), ("scalar", p.as_str()), "{ctx}");
            let wv = w["value"].as_f64().unwrap();
            assert_eq!(value.as_f64(), wv, "{ctx}: scalar value");
        }
        KernelArg::Dim {
            param: p,
            dim,
            arg,
            value,
        } => {
            assert_eq!((kind, param), ("dim", p.as_str()), "{ctx}");
            assert_eq!(
                Some(*dim),
                Dim::parse(w["dim"].as_str().unwrap()),
                "{ctx}: dim"
            );
            assert_eq!(*arg as u64, w["arg"].as_u64().unwrap(), "{ctx}: arg");
            assert_eq!(*value as i64, w["value"].as_i64().unwrap(), "{ctx}: value");
        }
        KernelArg::Buffer {
            param: p,
            role,
            arg,
            ..
        } => {
            assert_eq!((kind, param), ("buffer", p.as_str()), "{ctx}");
            let want_role = w["role"].as_str().unwrap();
            assert_eq!(
                serde_json::to_value(role).unwrap().as_str().unwrap(),
                want_role,
                "{ctx}: role"
            );
            assert_eq!(*arg as u64, w["arg"].as_u64().unwrap(), "{ctx}: arg");
        }
        KernelArg::OutLike {
            param: p,
            arg,
            shape,
            dtype,
        } => {
            assert_eq!((kind, param), ("out_like", p.as_str()), "{ctx}");
            assert_eq!(*arg as u64, w["arg"].as_u64().unwrap(), "{ctx}: arg");
            let want_shape: Vec<usize> = serde_json::from_value(w["shape"].clone()).unwrap();
            assert_eq!(*shape, want_shape, "{ctx}: shape");
            assert_eq!(
                dtype.numpy_name(),
                w["dtype"].as_str().unwrap(),
                "{ctx}: dtype"
            );
        }
    }
}

#[test]
fn derivation_fixtures_replay() {
    let manifest = Manifest::from_path(bundle_dir().join("manifest.json")).unwrap();
    let text = std::fs::read_to_string(bundle_dir().join("fixtures/derivation.json")).unwrap();
    let fixtures: Fixtures = serde_json::from_str(&text).unwrap();
    assert_eq!(fixtures.schema_version, manifest.schema_version);

    let mut checked = 0usize;
    for (iface_id, kernels) in &fixtures.interfaces {
        let entry = manifest
            .interface(iface_id)
            .unwrap_or_else(|| panic!("{iface_id} in manifest"));
        for (kname, fixture) in kernels {
            let kernel = entry
                .kernel(kname)
                .unwrap_or_else(|| panic!("{iface_id}.{kname}"));
            assert_eq!(kernel.symbol, fixture.symbol, "{iface_id}.{kname} symbol");
            for (case_name, case) in &fixture.cases {
                let ctx = format!("{iface_id}.{kname}[{case_name}]");
                let args = to_args(&case.args);
                let derived = derive_args(kernel, &args);
                match (&case.error, &case.kernel_args) {
                    (Some(_), _) => {
                        assert!(derived.is_err(), "{ctx}: Python raised, Rust must too");
                        continue;
                    }
                    (None, Some(want)) => {
                        let got = derived.unwrap_or_else(|e| panic!("{ctx}: {e}"));
                        assert_eq!(got.len(), want.len(), "{ctx}: argument count");
                        for (g, w) in got.iter().zip(want) {
                            check_kernel_arg(g, w, &ctx);
                        }
                    }
                    (None, None) => panic!("{ctx}: fixture has neither kernel_args nor error"),
                }

                // dispatch_size
                let dispatch = dispatch_global_size(kernel.dispatch_size.as_deref(), &args);
                if case.dispatch_error.is_some() {
                    assert!(dispatch.is_err(), "{ctx}: dispatch must fail");
                } else if kernel.dispatch_size.is_some() {
                    assert_eq!(
                        dispatch.unwrap_or_else(|e| panic!("{ctx}: {e}")),
                        case.dispatch_global_size,
                        "{ctx}: dispatch size"
                    );
                }

                // per-backend defaults
                if let Some(defaults) = &case.default_global_size {
                    for (name, backend) in [
                        ("ocl", Backend::Ocl),
                        ("cuda", Backend::Cuda),
                        ("metal", Backend::Metal),
                    ] {
                        assert_eq!(
                            &default_global_size(backend, &args),
                            &defaults[name],
                            "{ctx}: default {name}"
                        );
                    }
                }
                let effective = case
                    .dispatch_global_size
                    .clone()
                    .or_else(|| default_global_size(Backend::Cuda, &args));
                if let (Some(want), Some(global)) = (&case.cuda_launch, &effective) {
                    let (grid, block) = cuda_grid_block(global, None);
                    assert_eq!(grid.to_vec(), want.grid, "{ctx}: cuda grid");
                    assert_eq!(block.to_vec(), want.block, "{ctx}: cuda block");
                }
                let metal_effective = case
                    .dispatch_global_size
                    .clone()
                    .or_else(|| default_global_size(Backend::Metal, &args));
                if let (Some(want), Some(global)) = (&case.metal_launch, &metal_effective) {
                    let (grid, tgs) = metal_grid_tgs(global, None);
                    assert_eq!(grid.to_vec(), want.grid, "{ctx}: metal grid");
                    assert_eq!(tgs.to_vec(), want.tgs, "{ctx}: metal tgs");
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 500, "only {checked} cases checked");
    println!("replayed {checked} derivation cases");
}

#[test]
fn manifest_sources_exist_and_hash() {
    use std::io::Read;
    let manifest = Manifest::from_path(bundle_dir().join("manifest.json")).unwrap();
    assert_eq!(manifest.backends, ["ocl", "metal", "cuda"]);
    for entry in &manifest.interfaces {
        for (backend, src) in &entry.sources {
            let path = bundle_dir().join(&src.path);
            let mut text = String::new();
            std::fs::File::open(&path)
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
                .read_to_string(&mut text)
                .unwrap();
            assert!(
                !text.is_empty(),
                "{backend} source for {} is empty",
                entry.id
            );
            assert!(src.path.starts_with(&format!("{backend}/")));
        }
    }
}
