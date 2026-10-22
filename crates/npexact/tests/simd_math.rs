//! Replay the `simd_math` host fixtures: numpy's float32 sin/cos and
//! float32/float64 tanh on this host, bit for bit.

use npexact::simd_math::{cos_f32, sin_f32, tanh_f32, tanh_f64};
use serde::Deserialize;

mod common;

#[derive(Deserialize)]
struct Index {
    libm: std::collections::BTreeMap<String, bool>,
    cases: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
struct Case {
    func: String,
    name: String,
    file: String,
}

#[test]
fn simd_math_matches_numpy() {
    let Some(dir) = common::fixture_dir(env!("CARGO_MANIFEST_DIR")) else {
        return;
    };
    let index: Index =
        serde_json::from_str(&std::fs::read_to_string(dir.join("index.json")).unwrap()).unwrap();
    // The port's assumptions about which functions are libm on this host
    for (name, expect_libm) in [
        ("sin_f32", false),
        ("cos_f32", false),
        ("tanh_f32", false),
        ("tanh_f64", false),
        ("exp_f32", true),
        ("log_f32", true),
        ("sin_f64", true),
        ("cos_f64", true),
        ("exp_f64", true),
        ("log_f64", true),
    ] {
        assert_eq!(
            index.libm[name], expect_libm,
            "{name}: numpy == libm is {} on this host; the port assumes {expect_libm}",
            index.libm[name]
        );
    }
    let mut checked = 0;
    let cases: Vec<Case> = index
        .cases
        .iter()
        .filter(|c| {
            c["func"]
                .as_str()
                .is_some_and(|f| f.starts_with("simd_math."))
        })
        .map(|c| serde_json::from_value(c.clone()).unwrap())
        .collect();
    for case in &cases {
        let mut npz = common::open_npz(&dir.join(&case.file));
        let label = format!("{} {}", case.func, case.name);
        match case.func.as_str() {
            "simd_math.sin_f32" | "simd_math.cos_f32" | "simd_math.tanh_f32" => {
                let x = common::read_array::<f32>(&mut npz, "x");
                let want = common::read_array::<f32>(&mut npz, "y");
                let f: fn(f32) -> f32 = match case.func.as_str() {
                    "simd_math.sin_f32" => sin_f32,
                    "simd_math.cos_f32" => cos_f32,
                    _ => tanh_f32,
                };
                let got: Vec<f32> = x.iter().map(|&v| f(v)).collect();
                if let Some(i) = common::first_bit_diff_f32(&got, want.as_slice().unwrap()) {
                    panic!(
                        "{label}: element {i} (x={:?}): got {:?}, want {:?}",
                        x[i], got[i], want[i]
                    );
                }
            }
            "simd_math.tanh_f64" => {
                let x = common::read_array::<f64>(&mut npz, "x");
                let want = common::read_array::<f64>(&mut npz, "y");
                let got: Vec<f64> = x.iter().map(|&v| tanh_f64(v)).collect();
                if let Some(i) = common::first_bit_diff_f64(&got, want.as_slice().unwrap()) {
                    panic!(
                        "{label}: element {i} (x={:?}): got {:?}, want {:?}",
                        x[i], got[i], want[i]
                    );
                }
            }
            other => panic!("unknown case {other}"),
        }
        checked += 1;
    }
    println!("{checked} simd_math cases match numpy");
    assert!(checked >= 20);
}
