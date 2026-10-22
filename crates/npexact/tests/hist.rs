//! Replay the `hist` host fixtures: histogram, density, linspace, interp, cumsum
//! and VI's histeq chain, bit for bit.

use npexact::hist::*;
use npexact::random::Contraction;
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
}

fn bits64(a: &[f64], b: &[f64]) -> Option<usize> {
    common::first_bit_diff_f64(a, b)
}

#[test]
fn hist_matches_numpy() {
    let Some(dir) = common::fixture_dir(env!("CARGO_MANIFEST_DIR")) else {
        return;
    };
    let index: Index =
        serde_json::from_str(&std::fs::read_to_string(dir.join("index.json")).unwrap()).unwrap();
    let contraction = if index.fma.normal_fma {
        Contraction::Fma
    } else {
        Contraction::Plain
    };
    let case = index
        .cases
        .iter()
        .find(|c| c["func"] == "hist")
        .expect("hist case");
    let mut npz = common::open_npz(&dir.join(case["file"].as_str().unwrap()));
    let inputs: Vec<String> = case["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let mut checked = 0;
    for name in inputs.iter().filter(|n| !n.starts_with("interp")) {
        let a = common::read_array::<f32>(&mut npz, &format!("in.{name}"));
        let a = a.as_slice().unwrap();
        for nbins in [256usize, 64] {
            let (counts, edges) = histogram_f32(a, nbins);
            let want_counts = common::read_array::<i64>(&mut npz, &format!("counts{nbins}.{name}"));
            assert_eq!(
                counts,
                want_counts.as_slice().unwrap(),
                "{name} counts{nbins}"
            );
            let want_bins = common::read_array::<f32>(&mut npz, &format!("bins{nbins}.{name}"));
            if let Some(i) = common::first_bit_diff_f32(&edges, want_bins.as_slice().unwrap()) {
                panic!(
                    "{name} bins{nbins}: edge {i}: got {:?}, want {:?}",
                    edges[i], want_bins[i]
                );
            }
            let (density, _) = histogram_density_f32(a, nbins);
            let want = common::read_array::<f64>(&mut npz, &format!("hist{nbins}.{name}"));
            if let Some(i) = bits64(&density, want.as_slice().unwrap()) {
                panic!(
                    "{name} hist{nbins}: bin {i}: got {:?}, want {:?}",
                    density[i], want[i]
                );
            }
            checked += 3;
        }
        // histeq: cdf = cumsum(hist) / cdf[-1]; interp(a, bins[:-1], cdf) as float32
        let (density, edges) = histogram_density_f32(a, 256);
        let cdf = cumsum_f64(&density);
        let last = cdf[cdf.len() - 1];
        let cdf: Vec<f64> = cdf.iter().map(|v| v / last).collect();
        let want_cdf = common::read_array::<f64>(&mut npz, &format!("cdf.{name}"));
        if let Some(i) = bits64(&cdf, want_cdf.as_slice().unwrap()) {
            panic!("{name} cdf: {i}: got {:?}, want {:?}", cdf[i], want_cdf[i]);
        }
        let xp: Vec<f64> = edges[..256].iter().map(|&e| e as f64).collect();
        let x: Vec<f64> = a.iter().map(|&v| v as f64).collect();
        let eq: Vec<f32> = interp(&x, &xp, &cdf, contraction)
            .iter()
            .map(|&v| v as f32)
            .collect();
        let want = common::read_array::<f32>(&mut npz, &format!("histeq.{name}"));
        if let Some(i) = common::first_bit_diff_f32(&eq, want.as_slice().unwrap()) {
            panic!(
                "{name} histeq (contraction {contraction:?}): {i}: got {:?}, want {:?}",
                eq[i], want[i]
            );
        }
        checked += 2;
    }
    // interp alone
    let xp = common::read_array::<f32>(&mut npz, "in.interp_xp");
    let fp = common::read_array::<f32>(&mut npz, "in.interp_fp");
    let x = common::read_array::<f32>(&mut npz, "in.interp_x");
    let got = interp(
        &x.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &xp.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &fp.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        contraction,
    );
    let want = common::read_array::<f64>(&mut npz, "interp.f32");
    if let Some(i) = bits64(&got, want.as_slice().unwrap()) {
        panic!("interp.f32: {i}: got {:?}, want {:?}", got[i], want[i]);
    }
    let xp = common::read_array::<f64>(&mut npz, "in.interp64_xp");
    let fp = common::read_array::<f64>(&mut npz, "in.interp64_fp");
    let x = common::read_array::<f64>(&mut npz, "in.interp64_x");
    let got = interp(
        x.as_slice().unwrap(),
        xp.as_slice().unwrap(),
        fp.as_slice().unwrap(),
        contraction,
    );
    let want = common::read_array::<f64>(&mut npz, "interp.f64");
    if let Some(i) = bits64(&got, want.as_slice().unwrap()) {
        panic!("interp.f64: {i}: got {:?}, want {:?}", got[i], want[i]);
    }
    let want = common::read_array::<f32>(&mut npz, "linspace.f32");
    assert_eq!(linspace_f32(0.1, 0.9, 257), want.as_slice().unwrap());
    let want = common::read_array::<f32>(&mut npz, "linspace.f32b");
    assert_eq!(linspace_f32(-3.5, 12.25, 65), want.as_slice().unwrap());
    println!("{checked} hist checks match numpy");
}
