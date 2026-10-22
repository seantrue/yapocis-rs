//! Replay the `random` host fixtures (`yapocis-tools bundle --host-fixtures`):
//! each case seeds one generator and makes a sequence of draws; every output
//! must match numpy's bit for bit.

use npexact::random::{Contraction, Generator};
use serde::Deserialize;

mod common;

#[derive(Deserialize)]
struct Index {
    fma: Fma,
    cases: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
struct Fma {
    normal_fma: bool,
    normal_plain: bool,
    uniform_fma: bool,
}

#[derive(Deserialize)]
struct Case {
    #[allow(dead_code)]
    func: String,
    name: String,
    file: String,
    seed: Vec<String>, // decimal, since JSON floats would round big seeds
    calls: Vec<Call>,
}

#[derive(Deserialize)]
struct Call {
    method: String,
    args: serde_json::Value,
    out: String,
}

fn arg_u(args: &serde_json::Value, key: &str) -> usize {
    args[key].as_u64().unwrap_or_else(|| panic!("{key}")) as usize
}

fn arg_i(args: &serde_json::Value, key: &str) -> i64 {
    args[key].as_i64().unwrap_or_else(|| panic!("{key}"))
}

fn arg_f(args: &serde_json::Value, key: &str) -> f64 {
    args[key].as_f64().unwrap_or_else(|| panic!("{key}"))
}

enum Out {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I64(Vec<i64>),
}

fn run(g: &mut Generator, call: &Call) -> Out {
    let a = &call.args;
    match call.method.as_str() {
        "random" => {
            if a["dtype"].as_str() == Some("float32") {
                Out::F32(g.random_f32(arg_u(a, "size")))
            } else {
                Out::F64(g.random_f64(arg_u(a, "size")))
            }
        }
        "integers" => Out::I64(g.integers(arg_i(a, "low"), arg_i(a, "high"), arg_u(a, "size"))),
        "integers_broadcast" => {
            let highs: Vec<i64> = a["highs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            Out::I64(g.integers_broadcast(arg_i(a, "low"), &highs))
        }
        "choice" => Out::I64(g.choice_without_replacement(arg_u(a, "pop"), arg_u(a, "size"))),
        "permutation" => Out::I64(g.permutation(arg_u(a, "n"))),
        "shuffle_arange" => {
            let mut v: Vec<i64> = (0..arg_i(a, "n")).collect();
            g.shuffle(&mut v);
            Out::I64(v)
        }
        "standard_normal" => Out::F64(g.standard_normal(arg_u(a, "size"))),
        "standard_exponential" => Out::F64(g.standard_exponential(arg_u(a, "size"))),
        "normal" => Out::F64(g.normal(arg_f(a, "loc"), arg_f(a, "scale"), arg_u(a, "size"))),
        "uniform" => Out::F64(g.uniform(arg_f(a, "low"), arg_f(a, "high"), arg_u(a, "size"))),
        "exponential" => Out::F64(g.exponential(arg_f(a, "scale"), arg_u(a, "size"))),
        "geometric" => Out::I64(g.geometric(arg_f(a, "p"), arg_u(a, "size"))),
        other => panic!("unknown method {other}"),
    }
}

#[test]
fn random_draws_match_numpy() {
    let Some(dir) = common::fixture_dir(env!("CARGO_MANIFEST_DIR")) else {
        return;
    };
    let index: Index =
        serde_json::from_str(&std::fs::read_to_string(dir.join("index.json")).unwrap()).unwrap();
    assert_eq!(
        index.fma.normal_fma, index.fma.uniform_fma,
        "numpy contracted one but not the other"
    );
    assert!(
        index.fma.normal_fma || index.fma.normal_plain,
        "numpy's normal is neither fma nor plain"
    );
    let contraction = if index.fma.normal_fma {
        Contraction::Fma
    } else {
        Contraction::Plain
    };
    assert_eq!(
        contraction,
        Contraction::host_default(),
        "host default contraction policy is wrong for this host"
    );
    let mut checked = 0;
    let cases: Vec<Case> = index
        .cases
        .iter()
        .filter(|c| c["func"] == "random")
        .map(|c| serde_json::from_value(c.clone()).unwrap())
        .collect();
    for case in &cases {
        let mut npz = common::open_npz(&dir.join(&case.file));
        let seed: Vec<u128> = case.seed.iter().map(|s| s.parse().unwrap()).collect();
        let mut g = Generator::default_rng(&seed).with_contraction(contraction);
        for call in &case.calls {
            let label = format!(
                "{}: {}({}) -> {}",
                case.name, call.method, call.args, call.out
            );
            match run(&mut g, call) {
                Out::F64(got) => {
                    let want = common::read_array::<f64>(&mut npz, &call.out);
                    let want = want.as_slice().unwrap();
                    if let Some(i) = common::first_bit_diff_f64(&got, want) {
                        panic!(
                            "{label}: element {i}: got {:?}, want {:?}",
                            got.get(i),
                            want.get(i)
                        );
                    }
                }
                Out::F32(got) => {
                    let want = common::read_array::<f32>(&mut npz, &call.out);
                    let want = want.as_slice().unwrap();
                    if let Some(i) = common::first_bit_diff_f32(&got, want) {
                        panic!(
                            "{label}: element {i}: got {:?}, want {:?}",
                            got.get(i),
                            want.get(i)
                        );
                    }
                }
                Out::I64(got) => {
                    let want = common::read_array::<i64>(&mut npz, &call.out);
                    let want = want.as_slice().unwrap();
                    assert_eq!(got.len(), want.len(), "{label}: length");
                    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
                        panic!("{label}: element {i}: got {}, want {}", got[i], want[i]);
                    }
                }
            }
            checked += 1;
        }
    }
    println!("{checked} draw sequences match numpy");
    assert!(checked >= 20);
}
