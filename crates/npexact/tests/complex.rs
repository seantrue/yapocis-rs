//! Replay the `complex` host fixtures, bit for bit: numpy's complex64 multiply through
//! its ufunc loop (arrays) and its scalar path. Pure Rust, so every recorded host is
//! replayed anywhere, each under the rules for the machine and CPU it was recorded on.

use npexact::complex::{multiply_c64, Mul};
use std::fs::File;

mod common;

type Npz = npzfile::NpzReader<File>;

struct Case {
    host: String,
    array: Mul,
    scalar: Mul,
    npz: Npz,
    results: Vec<String>,
}

fn cases() -> Vec<Case> {
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
            .find(|c| c["func"] == "complex")
        else {
            eprintln!("{host}: no complex case (recorded before it existed)");
            continue;
        };
        let machine = index["machine"].as_str().unwrap_or("");
        let feat = &case["cpu_features"];
        let fma3_avx2 = feat["FMA3"] == true && feat["AVX2"] == true;
        out.push(Case {
            array: Mul::array_for(machine, fma3_avx2),
            scalar: Mul::scalar_for(machine),
            npz: common::open_npz(&dir.join(case["file"].as_str().unwrap())),
            results: case["results"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect(),
            host,
        });
    }
    assert!(!out.is_empty(), "no host has a complex case");
    out
}

fn c64s(npz: &mut Npz, name: &str) -> Vec<[f32; 2]> {
    let re = common::read_array::<f32>(npz, &format!("{name}.re"));
    let im = common::read_array::<f32>(npz, &format!("{name}.im"));
    assert_eq!(re.len(), im.len(), "{name}");
    re.iter().zip(im.iter()).map(|(&r, &i)| [r, i]).collect()
}

/// Bit patterns equal, any NaN matching any NaN (numpy doesn't promise a NaN's payload).
fn same(what: &str, got: &[[f32; 2]], want: &[[f32; 2]], a: &[[f32; 2]], b: &[[f32; 2]]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let bits = |v: f32| if v.is_nan() { u32::MAX } else { v.to_bits() };
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g.iter().map(|&v| bits(v)).ne(w.iter().map(|&v| bits(v))) {
            panic!(
                "{what}: element {i}: {:?} × {:?} = {g:?}, numpy {w:?}",
                a.get(i),
                b.get(i)
            );
        }
    }
}

#[test]
fn complex_multiply_matches_numpy() {
    for Case {
        host,
        array,
        scalar,
        mut npz,
        results,
    } in cases()
    {
        let (pa, pb) = (c64s(&mut npz, "in.pairs_a"), c64s(&mut npz, "in.pairs_b"));
        let (ra, rb) = (c64s(&mut npz, "in.rand_a"), c64s(&mut npz, "in.rand_b"));
        let mut checked = 0;
        let mut check = |name: &str, a: &[[f32; 2]], b: &[[f32; 2]], m: Mul| {
            let want = c64s(&mut npz, name);
            same(
                &format!("{host}: {name} ({m:?})"),
                &multiply_c64(m, a, b),
                &want,
                a,
                b,
            );
            checked += want.len();
        };
        check("mul.pairs", &pa, &pb, array);
        check("mul.rand", &ra, &rb, array);
        check("square.rand", &ra, &ra, array);
        for n in 1..=33 {
            check(&format!("mul.len{n}"), &ra[..n], &rb[..n], array);
        }
        let (sa, sb): (Vec<_>, Vec<_>) = (
            ra.iter().step_by(3).copied().collect(),
            rb.iter().step_by(3).copied().collect(),
        );
        check("mul.stride3", &sa, &sb, array);
        check("mul.by_scalar", &ra, &vec![rb[0]; ra.len()], array);
        // numpy scalars: the first 2,000 random pairs, then every 7th special pair
        let sa: Vec<_> = ra[..2000]
            .iter()
            .copied()
            .chain(pa.iter().step_by(7).copied())
            .collect();
        let sb: Vec<_> = rb[..2000]
            .iter()
            .copied()
            .chain(pb.iter().step_by(7).copied())
            .collect();
        check("mul.scalars", &sa, &sb, scalar);
        // the fixtures tell the rules apart: the other rule misses numpy somewhere
        let other = |m: Mul| {
            if m == Mul::Fused {
                Mul::Plain
            } else {
                Mul::Fused
            }
        };
        let want = c64s(&mut npz, "mul.rand");
        assert_ne!(
            multiply_c64(other(array), &ra, &rb),
            want,
            "{host}: mul.rand can't tell the rules apart"
        );
        let want = c64s(&mut npz, "mul.scalars");
        assert_ne!(
            multiply_c64(other(scalar), &sa, &sb),
            want,
            "{host}: mul.scalars can't tell the rules apart"
        );
        let named = results.iter().filter(|r| r.ends_with(".re")).count();
        assert_eq!(
            named,
            3 + 33 + 3,
            "{host}: a result this test doesn't check"
        );
        println!(
            "{host}: {checked} complex products match numpy (arrays {array:?}, scalars {scalar:?})"
        );
    }
}
