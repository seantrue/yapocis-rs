//! Replay the `reduce` host fixtures: numpy's sum/mean/var/std on this host,
//! bit for bit.

use npexact::reduce::*;
use serde::Deserialize;

mod common;

#[derive(Deserialize)]
struct Index {
    cases: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
struct Case {
    func: String,
    file: String,
    inputs: Vec<String>,
    results: Vec<String>,
}

fn check<T: Copy + std::fmt::Debug + PartialEq>(
    label: &str,
    got: T,
    want: T,
    bits: fn(T) -> u64,
    nan: fn(T) -> bool,
) {
    if bits(got) != bits(want) && !(nan(got) && nan(want)) {
        panic!("{label}: got {got:?}, want {want:?}");
    }
}

#[test]
fn reductions_match_numpy() {
    let Some(dir) = common::fixture_dir(env!("CARGO_MANIFEST_DIR")) else {
        return;
    };
    let index: Index =
        serde_json::from_str(&std::fs::read_to_string(dir.join("index.json")).unwrap()).unwrap();
    let mut checked = 0;
    let cases: Vec<Case> = index
        .cases
        .iter()
        .filter(|c| c["func"].as_str().is_some_and(|f| f.starts_with("reduce.")))
        .map(|c| serde_json::from_value(c.clone()).unwrap())
        .collect();
    for case in &cases {
        let mut npz = common::open_npz(&dir.join(&case.file));
        let f32s = case.func == "reduce.f32";
        for input in &case.inputs {
            for op in ["sum", "mean", "var0", "std0", "var1", "std1"] {
                let key = format!("{op}.{input}");
                if !case.results.contains(&key) {
                    continue;
                }
                let label = format!("{} {key}", case.func);
                if f32s {
                    let a = common::read_array::<f32>(&mut npz, &format!("in.{input}"));
                    let a = a.as_slice().unwrap();
                    let want = common::read_array::<f32>(&mut npz, &key)[[]];
                    let got = match op {
                        "sum" => sum_f32(a),
                        "mean" => mean_f32(a),
                        "var0" => var_f32(a, 0),
                        "std0" => std_f32(a, 0),
                        "var1" => var_f32(a, 1),
                        _ => std_f32(a, 1),
                    };
                    check(&label, got, want, |v| v.to_bits() as u64, |v| v.is_nan());
                } else {
                    let a = common::read_array::<f64>(&mut npz, &format!("in.{input}"));
                    let a = a.as_slice().unwrap();
                    let want = common::read_array::<f64>(&mut npz, &key)[[]];
                    let got = match op {
                        "sum" => sum_f64(a),
                        "mean" => mean_f64(a),
                        "var0" => var_f64(a, 0),
                        "std0" => std_f64(a, 0),
                        "var1" => var_f64(a, 1),
                        _ => std_f64(a, 1),
                    };
                    check(&label, got, want, |v| v.to_bits(), |v| v.is_nan());
                }
                checked += 1;
            }
        }
    }
    println!("{checked} reductions match numpy");
    assert!(checked >= 1000);
}
