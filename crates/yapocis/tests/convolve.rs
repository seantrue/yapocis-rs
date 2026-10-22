//! Runtime-weight convolutions (`Convolve1D`, `Gaussian::with_weights`) against
//! a CPU reference, the bundled variant, and the Python parity fixtures.

mod common;

use ndarray::{Array1, Array2};
use yapocis::interfaces::convolve_box3::ConvolveBox3;
use yapocis::lib::convolve::Convolve1D;
use yapocis::lib::gaussian::{default_weights, gaussian_weight, Gaussian};

fn cpu_convolve(a: &[f32], w: &[f64]) -> Vec<f32> {
    let left = w.len() / 2;
    let right = a.len() - left;
    (0..a.len())
        .map(|i| {
            if i > left && i < right {
                w.iter()
                    .enumerate()
                    .map(|(j, wj)| a[(i as i64 + j as i64 - left as i64) as usize] * *wj as f32)
                    .sum()
            } else {
                0.0
            }
        })
        .collect()
}

#[test]
fn custom_weights_match_cpu_and_bundled_variant() {
    for device in common::devices() {
        let signal: Vec<f32> = (0..2048).map(|i| ((i * 7) % 13) as f32 / 13.0).collect();
        let a = Array1::from_vec(signal.clone()).into_dyn();

        let w7 = [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05];
        let cnv = Convolve1D::new(&device, "k7", &w7).unwrap();
        let out = cnv.convolve(&a.view()).unwrap();
        let want = cpu_convolve(&signal, &w7);
        common::allclose_f32(
            out.as_slice().unwrap(),
            &want,
            common::tolerances(1e-5, 1e-5).0,
            common::tolerances(1e-5, 1e-5).1,
        )
        .unwrap_or_else(|e| panic!("{device:?}: {e}"));

        // Same weights as the bundled variant: identical source, identical output.
        let box3 = [0.3333333333, 0.3333333333, 0.3333333333];
        let generated = Convolve1D::new(&device, "box3", &box3).unwrap();
        let bundled = ConvolveBox3::load(&device).unwrap();
        let g = generated.convolve(&a.view()).unwrap();
        let b = bundled.box3(&a.view()).unwrap();
        assert_eq!(g, b, "{device:?}: generated vs bundled box3");
    }
}

#[test]
fn gaussian_with_default_ladder_equals_bundled() {
    for device in common::devices() {
        let img = Array2::from_shape_fn((64, 48), |(x, y)| ((x * 3 + y * 5) % 17) as f32 / 17.0)
            .into_dyn();
        let bundled = Gaussian::load(&device).unwrap();
        let custom = Gaussian::with_weights(&device, &default_weights()).unwrap();
        assert_eq!(bundled.scales(), custom.scales());
        for scale in 0..3 {
            let a = bundled.gauss_image(&img.view(), scale).unwrap();
            let b = custom.gauss_image(&img.view(), scale).unwrap();
            assert_eq!(a, b, "{device:?}: scale {scale}");
        }
    }
}

#[test]
fn parity_with_python_runtime_weights() {
    let mut any = false;
    for device in common::devices() {
        let Some(index) = common::lib_index(device.backend()) else {
            continue;
        };
        let Some(cases) = index["generators"].get("convolve") else {
            continue;
        };
        let rtol = index["rtol"].as_f64().unwrap();
        let atol = index["atol"].as_f64().unwrap();
        let dir = common::parity_lib_dir(device.backend());
        for case in cases.as_array().unwrap() {
            any = true;
            let mut npz = common::open_npz(&dir.join(case["file"].as_str().unwrap()));
            match case["name"].as_str().unwrap() {
                "k7" => {
                    let weights: Vec<f64> =
                        serde_json::from_value(case["weights"].clone()).unwrap();
                    let signal = common::read_any(&mut npz, "signal").to_array::<f32>();
                    let want = common::read_any(&mut npz, "out");
                    let cnv = Convolve1D::new(&device, "k7", &weights).unwrap();
                    let got = cnv.convolve(&signal.view()).unwrap();
                    common::allclose_f32(
                        got.as_slice().unwrap(),
                        want.as_slice::<f32>(),
                        rtol,
                        atol,
                    )
                    .unwrap_or_else(|e| panic!("{device:?} k7: {e}"));
                }
                "gaussian_custom" => {
                    let weights: Vec<Vec<f64>> =
                        serde_json::from_value(case["weights"].clone()).unwrap();
                    let image = common::read_any(&mut npz, "image").to_array::<f32>();
                    let g = Gaussian::with_weights(&device, &weights).unwrap();
                    for (scale, key) in [(0usize, "blur0"), (1, "blur1")] {
                        let want = common::read_any(&mut npz, key);
                        let got = g.gauss_image(&image.view(), scale).unwrap();
                        common::allclose_f32(
                            got.as_slice().unwrap(),
                            want.as_slice::<f32>(),
                            rtol,
                            atol,
                        )
                        .unwrap_or_else(|e| panic!("{device:?} {key}: {e}"));
                    }
                    let (zca, grad, theta) = g.zcsdog(&image.view(), 0, None, true, true).unwrap();
                    for (name, got) in [("zca", zca), ("grad", grad), ("theta", theta)] {
                        let want = common::read_any(&mut npz, name);
                        common::allclose_f32(
                            got.as_slice().unwrap(),
                            want.as_slice::<f32>(),
                            rtol,
                            atol,
                        )
                        .unwrap_or_else(|e| panic!("{device:?} {name}: {e}"));
                    }
                }
                other => panic!("unknown case {other}"),
            }
        }
    }
    if !any {
        eprintln!("no convolve parity fixtures found");
    }
}

#[test]
fn weight_generator_sigma_ladder() {
    let w = gaussian_weight(2.5, 1.0, 0.001, true);
    assert!(w.len() > 9);
    assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
}
