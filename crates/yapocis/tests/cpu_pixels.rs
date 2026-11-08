//! The CPU ports of the pixel kernels the parity fixtures do not cover
//! (`hsi.rgb2hsi`/`hsi2rgb`, `audio.fused_vad_features`, `colormath` on
//! more transforms) against OpenCL and Metal on this machine. Skips when the
//! CPU backend or both GPU backends are absent.

use ndarray::{Array1, Array2, ArrayD};
use yapocis::interfaces::audio::Audio;
use yapocis::interfaces::colormath::Colormath;
use yapocis::interfaces::hsi::Hsi;
use yapocis::{available_backends, Backend, Device};

mod common;

/// The CPU device and every GPU reference that opens here.
fn devices() -> Option<(Device, Vec<Device>)> {
    let backends = available_backends();
    if !backends.contains(&Backend::RustCpu) {
        eprintln!("skipping: the CPU backend is not compiled in");
        return None;
    }
    let cpu = Device::open(Backend::RustCpu, None).unwrap();
    let refs: Vec<Device> = backends
        .into_iter()
        .filter(|b| matches!(b, Backend::Ocl | Backend::Metal))
        .filter_map(|b| Device::open(b, None).ok())
        .collect();
    if refs.is_empty() {
        eprintln!("skipping: no OpenCL or Metal device to compare with");
        return None;
    }
    Some((cpu, refs))
}

fn close(label: &str, got: &ArrayD<f32>, want: &ArrayD<f32>, tol: f64) {
    assert_eq!(got.shape(), want.shape(), "{label}");
    common::allclose_f32(got.as_slice().unwrap(), want.as_slice().unwrap(), tol, tol)
        .unwrap_or_else(|e| panic!("{label}: {e}"));
}

/// Deterministic values in `[0, 1)` (a 64-bit LCG, top 24 bits).
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect()
}

#[test]
fn planar_hsi_matches_gpu() {
    let Some((cpu, refs)) = devices() else { return };
    let n = 4096;
    let [r, g, b] = [1, 2, 3].map(|s| Array1::from_vec(values(n, s)).into_dyn());
    let c = Hsi::load(&cpu).unwrap();
    let (h, s, i) = c.rgb2hsi(&r.view(), &g.view(), &b.view()).unwrap();
    let back = c.hsi2rgb(&h.view(), &s.view(), &i.view()).unwrap();
    for device in refs {
        let k = Hsi::load(&device).unwrap();
        let (wh, ws, wi) = k.rgb2hsi(&r.view(), &g.view(), &b.view()).unwrap();
        for (name, got, want) in [("h", &h, &wh), ("s", &s, &ws), ("i", &i, &wi)] {
            close(&format!("{device:?} rgb2hsi {name}"), got, want, 1e-5);
        }
        let wback = k.hsi2rgb(&h.view(), &s.view(), &i.view()).unwrap();
        for (name, got, want) in [
            ("r", &back.0, &wback.0),
            ("g", &back.1, &wback.1),
            ("b", &back.2, &wback.2),
        ] {
            close(&format!("{device:?} hsi2rgb {name}"), got, want, 1e-5);
        }
    }
}

#[test]
fn colormath_transforms_match_gpu() {
    let Some((cpu, refs)) = devices() else { return };
    let pixels = Array2::from_shape_vec((1024, 3), values(3 * 1024, 7))
        .unwrap()
        .into_dyn();
    let c = Colormath::load(&cpu).unwrap();
    for device in refs {
        let k = Colormath::load(&device).unwrap();
        for id in [0, 13, 27, 45] {
            let opts = yapocis::rt::CallOptions {
                global: Some(vec![1024]),
                ..Default::default()
            };
            let got = yapocis::host::one_output::<f32>(
                c.transform_with(&opts, &pixels.view(), id).unwrap(),
            )
            .unwrap();
            let want = yapocis::host::one_output::<f32>(
                k.transform_with(&opts, &pixels.view(), id).unwrap(),
            )
            .unwrap();
            close(&format!("{device:?} transform {id}"), &got, &want, 1e-5);
        }
    }
}

#[test]
fn fused_vad_features_match_gpu() {
    let Some((cpu, refs)) = devices() else { return };
    let (height, width) = (64usize, 300usize);
    let spec = Array2::from_shape_vec((height, width), values(height * width, 11))
        .unwrap()
        .into_dyn();
    let run = |device: &Device, lower: i32, upper: i32| {
        let mut features = Array1::<f32>::zeros(width).into_dyn();
        Audio::load(device)
            .unwrap()
            .fused_vad_features(
                &spec.view(),
                lower,
                upper,
                width as i32,
                height as i32,
                &mut features.view_mut(),
            )
            .unwrap();
        features
    };
    for (lower, upper) in [(4, 40), (-3, 70), (10, 10)] {
        let got = run(&cpu, lower, upper);
        for device in &refs {
            close(
                &format!("{device:?} band {lower}..{upper}"),
                &got,
                &run(device, lower, upper),
                1e-4,
            );
        }
    }
}
