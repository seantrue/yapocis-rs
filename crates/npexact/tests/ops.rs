//! Replay the `elementwise`, `extrema`, `order` and `fft` host fixtures, bit for
//! bit: the numpy behaviours PlumCraft's array engine reproduces.

use npexact::{elementwise::*, extrema::*, fft::*, order::*};
use std::fs::File;

mod common;

type Npz = npzfile::NpzReader<File>;
type Binary = fn(MinMax, f32, f32) -> f32;
type Unary = fn(MinMax, f32) -> f32;
type Reduction = fn(MinMax, &[f32]) -> Option<f32>;

/// One host's fixtures for a section.
struct Case {
    host: String,
    policy: MinMax,
    npz: Npz,
    inputs: Vec<String>,
    results: Vec<String>,
}

/// The `func` case of **every** recorded host. These functions are pure Rust
/// (only `MinMax` follows the machine numpy ran on, and the index names it), so
/// each host's fixtures can be replayed anywhere. A host recorded before the
/// section existed is skipped with a note.
fn cases(func: &str) -> Vec<Case> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/host");
    let mut dirs: Vec<_> = std::fs::read_dir(&root)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|d| d.join("index.json").exists())
        .collect();
    dirs.sort();
    let mut out = Vec::new();
    for dir in dirs {
        let index: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("index.json")).unwrap())
                .unwrap();
        let host = index["host"].as_str().unwrap_or("?").to_string();
        let Some(case) = index["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["func"] == func)
        else {
            eprintln!("{host}: no {func} case (recorded before it existed)");
            continue;
        };
        let names = |k: &str| -> Vec<String> {
            case[k]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect()
        };
        out.push(Case {
            policy: MinMax::for_machine(index["machine"].as_str().unwrap_or("")),
            npz: common::open_npz(&dir.join(case["file"].as_str().unwrap())),
            inputs: names("inputs"),
            results: names("results"),
            host,
        });
    }
    assert!(!out.is_empty(), "no host has a {func} case");
    out
}

fn f32s(npz: &mut Npz, name: &str) -> Vec<f32> {
    common::read_array::<f32>(npz, name)
        .iter()
        .copied()
        .collect()
}

fn f64s(npz: &mut Npz, name: &str) -> Vec<f64> {
    common::read_array::<f64>(npz, name)
        .iter()
        .copied()
        .collect()
}

fn i64s(npz: &mut Npz, name: &str) -> Vec<i64> {
    common::read_array::<i64>(npz, name)
        .iter()
        .copied()
        .collect()
}

#[track_caller]
fn same_f32(what: &str, got: &[f32], want: &[f32], inputs: &[f32]) {
    if let Some(i) = common::first_bit_diff_f32(got, want) {
        panic!(
            "{what}: element {i} (input {:?}): got {:?} ({:#010x}), want {:?} ({:#010x})",
            inputs.get(i),
            got.get(i),
            got.get(i).map_or(0, |v| v.to_bits()),
            want.get(i),
            want.get(i).map_or(0, |v| v.to_bits())
        );
    }
}

#[test]
fn elementwise_matches_numpy() {
    for Case {
        host,
        policy,
        mut npz,
        results,
        ..
    } in cases("elementwise")
    {
        let mut checked = 0;
        for tag in ["pairs", "rand"] {
            let a = f32s(&mut npz, &format!("in.{tag}_a"));
            let b = f32s(&mut npz, &format!("in.{tag}_b"));
            let binary: [(&str, Binary); 3] = [
                ("minimum", minimum_f32_with),
                ("maximum", maximum_f32_with),
                ("mod", |_, x, y| mod_f32(x, y)),
            ];
            for (name, f) in binary {
                let got: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| f(policy, x, y)).collect();
                let want = f32s(&mut npz, &format!("{name}.{tag}"));
                same_f32(&format!("{host} {name}.{tag}"), &got, &want, &a);
                checked += 1;
            }
            let unary: [(&str, Unary); 3] = [
                ("sign", |_, v| sign_f32(v)),
                ("nan_to_num", |_, v| nan_to_num_f32(v)),
                ("clip0", |p, v| clip_min_f32_with(p, v, 0.0)),
            ];
            for (name, f) in unary {
                let got: Vec<f32> = a.iter().map(|&x| f(policy, x)).collect();
                let want = f32s(&mut npz, &format!("{name}.{tag}"));
                same_f32(&format!("{host} {name}.{tag}"), &got, &want, &a);
                checked += 1;
            }
            let turns = f32s(&mut npz, "in.turns");
            for (i, &t) in turns.iter().enumerate() {
                let key = format!("mod_scalar{i}.{tag}");
                if !results.contains(&key) {
                    continue;
                }
                let got: Vec<f32> = a.iter().map(|&x| mod_f32(x, t)).collect();
                let want = f32s(&mut npz, &key);
                same_f32(&format!("{host} {key} (turn {t})"), &got, &want, &a);
                checked += 1;
            }
        }
        println!("{host}: {checked} elementwise checks match numpy");
    }
}

#[test]
fn extrema_match_numpy() {
    for Case {
        host,
        policy,
        mut npz,
        inputs,
        ..
    } in cases("extrema")
    {
        let mut checked = 0;
        for name in inputs
            .iter()
            .filter(|n| !matches!(n.as_str(), "counts" | "wrap"))
        {
            let a = f32s(&mut npz, &format!("in.{name}"));
            let funcs: [(&str, Reduction); 4] = [
                ("min", min_f32_with),
                ("max", max_f32_with),
                ("nanmin", nanmin_f32_with),
                ("nanmax", nanmax_f32_with),
            ];
            for (f, func) in funcs {
                let want = f32s(&mut npz, &format!("{f}.{name}"));
                let got = func(policy, &a).expect("non-empty");
                same_f32(
                    &format!("{host} {f}.{name} ({policy:?})"),
                    &[got],
                    &want,
                    &[],
                );
                checked += 1;
            }
        }
        for name in ["counts", "wrap"] {
            let a = i64s(&mut npz, &format!("in.{name}"));
            assert_eq!(
                cumsum_i64(&a),
                i64s(&mut npz, &format!("cumsum_i64.{name}")),
                "{host} cumsum {name}"
            );
            checked += 1;
        }
        println!("{host}: {checked} extrema checks match numpy");
    }
}

#[test]
fn order_matches_numpy() {
    for Case {
        host,
        mut npz,
        inputs,
        results,
        ..
    } in cases("order")
    {
        let mut checked = 0;
        for name in inputs.iter().filter(|n| n.starts_with("sort")) {
            let a = f32s(&mut npz, &format!("in.{name}"));
            // numpy's arrangement of -0 and +0 depends on its sort algorithm (src/order.rs):
            // the values must match with the zeros' signs ignored
            let unsign = |v: Vec<f32>| -> Vec<f32> {
                v.into_iter()
                    .map(|x| if x == 0.0 { 0.0 } else { x })
                    .collect()
            };
            let want = unsign(f32s(&mut npz, &format!("sort.{name}")));
            same_f32(&format!("sort.{name}"), &unsign(sort_f32(&a)), &want, &[]);
            checked += 1;
        }
        let hay = f64s(&mut npz, "in.hay");
        let needles = f64s(&mut npz, "in.needles");
        let left: Vec<i64> = needles
            .iter()
            .map(|&v| searchsorted_left_f64(&hay, v) as i64)
            .collect();
        let right: Vec<i64> = needles
            .iter()
            .map(|&v| searchsorted_right_f64(&hay, v) as i64)
            .collect();
        assert_eq!(left, i64s(&mut npz, "searchsorted_left"));
        assert_eq!(right, i64s(&mut npz, "searchsorted_right"));
        let keys: Vec<Vec<f32>> = (0..4)
            .map(|i| f32s(&mut npz, &format!("in.lexkey{i}")))
            .collect();
        let key_refs: Vec<&[f32]> = keys.iter().map(Vec::as_slice).collect();
        let got: Vec<i64> = lexsort_f32(&key_refs)
            .unwrap()
            .into_iter()
            .map(|i| i as i64)
            .collect();
        assert_eq!(got, i64s(&mut npz, "lexsort"), "lexsort");
        checked += 3;
        let qs = f64s(&mut npz, "in.qs");
        let pairs = f64s(&mut npz, "in.pct_pairs");
        for name in inputs.iter().filter(|n| n.starts_with("stat")) {
            let a = f32s(&mut npz, &format!("in.{name}"));
            let got: Vec<f32> = qs.iter().map(|&q| quantile_linear_f32(&a, q)).collect();
            same_f32(
                &format!("quantile.{name}"),
                &got,
                &f32s(&mut npz, &format!("quantile.{name}")),
                &[],
            );
            same_f32(
                &format!("median.{name}"),
                &[median_f32(&a)],
                &f32s(&mut npz, &format!("median.{name}")),
                &[],
            );
            checked += 2;
        }
        for name in inputs.iter().filter(|n| n.starts_with("nanstat")) {
            let a = f32s(&mut npz, &format!("in.{name}"));
            for (j, pair) in pairs.chunks(2).enumerate() {
                let key = format!("nanpercentile{j}.{name}");
                if !results.contains(&key) {
                    continue;
                }
                let got = nanpercentile_linear_f32(&a, pair);
                // all-NaN input: numpy returns its NaNs as float32
                let want = npz
                    .by_name::<ndarray::OwnedRepr<f64>, ndarray::IxDyn>(&key)
                    .map(|a| a.iter().copied().collect::<Vec<f64>>())
                    .unwrap_or_else(|_| f32s(&mut npz, &key).into_iter().map(f64::from).collect());
                if let Some(i) = common::first_bit_diff_f64(&got, &want) {
                    panic!("{key} {pair:?}: {i}: got {:?}, want {:?}", got[i], want[i]);
                }
                checked += 1;
            }
        }
        for name in inputs.iter().filter_map(|n| n.strip_prefix("uniq_")) {
            let a = i64s(&mut npz, &format!("in.uniq_{name}"));
            let (values, inverse) = unique_inverse_i64(&a);
            assert_eq!(
                values,
                i64s(&mut npz, &format!("unique.{name}")),
                "unique {name}"
            );
            let inverse: Vec<i64> = inverse.into_iter().map(|i| i as i64).collect();
            assert_eq!(
                inverse,
                i64s(&mut npz, &format!("unique_inverse.{name}")),
                "unique inverse {name}"
            );
            checked += 2;
        }
        println!("{host}: {checked} order checks match numpy");
    }
}

#[test]
fn fft_shifts_match_numpy() {
    for Case {
        mut npz, inputs, ..
    } in cases("fft")
    {
        for name in &inputs {
            let arr = common::read_array::<f32>(&mut npz, &format!("in.{name}"));
            let (h, w) = (arr.shape()[0], arr.shape()[1]);
            let a: Vec<f32> = arr.iter().copied().collect();
            for (f, func) in [
                (
                    "fftshift",
                    fftshift_2d::<f32> as fn(&[f32], usize, usize) -> Option<Vec<f32>>,
                ),
                ("ifftshift", ifftshift_2d::<f32>),
            ] {
                let want = f32s(&mut npz, &format!("{f}.{name}"));
                same_f32(&format!("{f}.{name}"), &func(&a, h, w).unwrap(), &want, &[]);
            }
        }
    }
}
