//! The library ports on every available backend.

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex;
use yapocis::lib::fft::Window;
use yapocis::lib::{Anisodiff, Colormath, Fft, Gaussian, Hsi, Median, Zcs};
use yapocis::{available_backends, Device};

fn devices() -> Vec<Device> {
    available_backends()
        .into_iter()
        .filter_map(|b| match Device::open(b, None) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("skipping {b:?}: {e}");
                None
            }
        })
        .collect()
}

fn test_image(w: usize, h: usize) -> Array2<f32> {
    Array2::from_shape_fn((w, h), |(x, y)| {
        let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
        if (fx - 0.5).powi(2) + (fy - 0.5).powi(2) < 0.09 {
            0.9
        } else {
            0.1 + 0.05 * (x % 3) as f32
        }
    })
}

#[test]
fn median_iterations_converge() {
    for device in devices() {
        let median = Median::load(&device).unwrap();
        let mut img = test_image(64, 48);
        img[[10, 10]] = 5.0; // salt
        let once = median.median3x3(&img.view().into_dyn(), 1).unwrap();
        assert!(once[[10, 10]] < 1.0, "{device:?}: salt removed");
        let five = median.median3x3(&img.view().into_dyn(), 5).unwrap();
        let again = median.median3x3(&five.view(), 1).unwrap();
        let diff: f32 = five
            .iter()
            .zip(again.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff < 1.0,
            "{device:?}: near fixed point after 5 passes (diff {diff})"
        );
        assert_eq!(
            median.median3x3(&img.view().into_dyn(), 0).unwrap(),
            img.into_dyn()
        );
    }
}

#[test]
fn anisodiff_smooths_and_preserves_shape() {
    for device in devices() {
        let aniso = Anisodiff::load(&device).unwrap();
        let img = test_image(48, 40).into_dyn();
        let out = aniso.anisodiff_n(&img.view(), 5, 3.0, 0.2).unwrap();
        assert_eq!(out.shape(), img.shape(), "{device:?}");
        let var = |a: &ndarray::ArrayD<f32>| {
            let m = a.mean().unwrap();
            a.iter().map(|v| (v - m).powi(2)).sum::<f32>() / a.len() as f32
        };
        assert!(
            var(&out) < var(&img),
            "{device:?}: diffusion reduces variance"
        );
        // 3-plane image goes through the padded 4-plane kernel and comes back as 3 planes
        let rgb = Array3::from_shape_fn((32, 24, 3), |(x, y, c)| {
            test_image(32, 24)[[x, y]] * (1.0 + c as f32 * 0.1)
        })
        .into_dyn();
        let out3 = aniso.anisodiff_n(&rgb.view(), 2, 3.0, 0.2).unwrap();
        assert_eq!(out3.shape(), &[32, 24, 3], "{device:?}");
        assert!(
            aniso.anisodiff_n(&img.view(), 1, 3.0, 0.3).is_err(),
            "step size is validated"
        );
    }
}

#[test]
fn gaussian_blur_and_zcsdog() {
    for device in devices() {
        let g = Gaussian::load(&device).unwrap();
        assert!(g.scales() >= 3, "{device:?}: {} scales", g.scales());
        assert_eq!(g.width(0).unwrap(), 9);
        let img = test_image(96, 80).into_dyn();
        let blurred = g.gauss_image(&img.view(), 1).unwrap();
        assert_eq!(blurred.shape(), img.shape());
        let edge_before = (img[[76, 40]] - img[[77, 40]]).abs();
        let edge_after = (blurred[[76, 40]] - blurred[[77, 40]]).abs();
        assert!(
            edge_after < edge_before,
            "{device:?}: blur softens the disc edge"
        );
        let (zca, grad, theta) = g.zcsdog(&img.view(), 0, None, true, true).unwrap();
        assert_eq!(zca.dim(), (96, 80));
        assert_eq!(grad.dim(), (96, 80));
        assert_eq!(theta.dim(), (96, 80));
        assert_eq!(zca[[0, 0]], 1.0, "frame");
        assert_eq!(zca[[5, 40]], 0.0, "margin cleared");
        let crossings = zca.iter().filter(|&&v| v != 0.0).count();
        assert!(
            crossings > 80,
            "{device:?}: zero crossings found around the disc ({crossings})"
        );
    }
}

#[test]
fn zcs_by_depth() {
    for device in devices() {
        let zcs = Zcs::load(&device).unwrap();
        let img = Array2::from_shape_fn((32, 32), |(x, _)| x as f32 - 15.5).into_dyn();
        let out = zcs.zcs(&img.view(), 0.0001).unwrap();
        assert_eq!(out.shape(), &[32, 32]);
        let two = Array3::from_shape_fn((32, 32, 2), |(x, _, _)| x as f32 - 15.5).into_dyn();
        let out2 = zcs.zcs(&two.view(), 0.0001).unwrap();
        assert_eq!(out2.shape(), &[32, 32, 2], "{device:?}");
    }
}

#[test]
fn hsi_and_colormath_per_pixel_sizes() {
    for device in devices() {
        let hsi = Hsi::load(&device).unwrap();
        let rgb = Array3::from_shape_fn((16, 12, 3), |(x, y, c)| {
            ((x * 7 + y * 3 + c * 5) % 10) as f32 / 10.0
        })
        .into_dyn();
        let h = hsi.rgb2hsi3(&rgb.view()).unwrap();
        let back = hsi.hsi2rgb3(&h.view()).unwrap();
        for (a, b) in rgb.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-3, "{device:?}: {a} vs {b}");
        }
        let cm = Colormath::load(&device).unwrap();
        let xyz = cm.transform_named("srgb_d65.rgb2xyz", &rgb.view()).unwrap();
        let rgb2 = cm.transform_named("srgb_d65.xyz2rgb", &xyz.view()).unwrap();
        for (a, b) in rgb.iter().zip(rgb2.iter()) {
            assert!((a - b).abs() < 1e-3, "{device:?}: {a} vs {b}");
        }
        let unity = cm.transform_named("unity", &rgb.view()).unwrap();
        assert_eq!(unity, rgb);
        assert!(cm.transform_named("nope", &rgb.view()).is_err());
    }
}

#[test]
fn fft_roundtrip_and_peak() {
    for device in devices() {
        let fft = Fft::load(&device).unwrap();
        let n = 256usize;
        let sample_rate = 1000.0;
        let signal =
            Array1::from_iter((0..n).map(|i| {
                (2.0 * std::f32::consts::PI * 100.0 * i as f32 / sample_rate as f32).sin()
            }));
        let complex: Array1<Complex<f32>> = signal.mapv(|v| Complex::new(v, 0.0));
        let spectrum = fft.forward(&complex.view()).unwrap();
        let back = fft.inverse(&spectrum.view()).unwrap();
        for (a, b) in complex.iter().zip(back.iter()) {
            assert!((a - b).norm() < 1e-3, "{device:?}: roundtrip {a} vs {b}");
        }
        let (peaks, _) = fft
            .find_peaks(&signal.view(), sample_rate, 1, Window::Hann)
            .unwrap();
        assert!(
            (peaks[0] - 100.0).abs() < 4.0,
            "{device:?}: peak at {} Hz",
            peaks[0]
        );
        // batch of identical signals equals the single FFT. The Python package
        // gets this wrong on Apple's OpenCL too (claude-debug/BATCH_FFT_CROSSBACKEND_STATUS.md),
        // so the check is skipped there.
        let batch = ndarray::stack(ndarray::Axis(0), &[complex.view(), complex.view()]).unwrap();
        let spectra = fft.batch_forward(&batch.view()).unwrap();
        assert_eq!(spectra.dim(), (2, n));
        if device.backend() != yapocis::Backend::Ocl {
            for (a, b) in spectra.row(1).iter().zip(spectrum.iter()) {
                assert!(
                    (a - b).norm() < 1e-2 * (1.0 + b.norm()),
                    "{device:?}: batch {a} vs {b}"
                );
            }
        }
        let power = fft.power_spectrum(&spectrum.view()).unwrap();
        assert_eq!(power.len(), n);
    }
}
