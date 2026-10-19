//! The generated typed API on every available backend.

use ndarray::{Array1, Array2, Array3};
use yapocis::interfaces::demo::Demo;
use yapocis::interfaces::gradient::Gradient;
use yapocis::interfaces::hsi::Hsi;
use yapocis::interfaces::mandelbrot::Mandelbrot;
use yapocis::interfaces::operators::Operators;
use yapocis::{available_backends, Backend, Device};

fn devices() -> Vec<Device> {
    available_backends()
        .into_iter()
        .filter_map(|b: Backend| match Device::open(b, None) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("skipping {b:?}: {e}");
                None
            }
        })
        .collect()
}

#[test]
fn demo_sum_typed() {
    for device in devices() {
        let demo = Demo::load(&device).unwrap();
        let a = Array1::from_vec((0..1000).map(|i| i as f32).collect()).into_dyn();
        let b = Array1::from_vec((0..1000).map(|i| (1000 - i) as f32).collect()).into_dyn();
        let c = demo.sum(&a.view(), &b.view()).unwrap();
        assert_eq!(c.shape(), &[1000]);
        assert!(c.iter().all(|&v| v == 1000.0), "{device:?}");

        // sum_res writes into the caller's array
        let mut c2 = Array1::<f32>::zeros(1000).into_dyn();
        demo.sum_res(&a.view(), &b.view(), &mut c2.view_mut())
            .unwrap();
        assert!(c2.iter().all(|&v| v == 1000.0), "{device:?}");

        // non-contiguous input is copied to standard layout transparently
        let big = Array2::from_shape_fn((10, 100), |(i, j)| (i * 100 + j) as f32);
        let col = big.t();
        let col_dyn = col.into_dyn();
        let s = demo.sum(&col_dyn, &col_dyn).unwrap();
        assert_eq!(s.shape(), &[100, 10]);
        assert_eq!(s[[3, 2]], 2.0 * big[[2, 3]]);
    }
}

#[test]
fn operators_variants_and_resident() {
    for device in devices() {
        let ops = Operators::load(&device).unwrap();
        let a = Array2::from_shape_fn((16, 8), |(i, j)| (i * 8 + j) as f32).into_dyn();
        let b = Array2::from_elem((16, 8), 2.0f32).into_dyn();
        let m = ops.mul(&a.view(), &b.view()).unwrap();
        assert_eq!(m[[5, 3]], 2.0 * a[[5, 3]], "{device:?}");
        let d = ops.div(&a.view(), &b.view()).unwrap();
        assert_eq!(d[[5, 3]], a[[5, 3]] / 2.0, "{device:?}");
        // resident extension: all three on the device
        let a_dev = device.upload(a.as_slice().unwrap(), a.shape()).unwrap();
        let b_dev = device.upload(b.as_slice().unwrap(), b.shape()).unwrap();
        let out = device.alloc::<f32>(a.shape()).unwrap();
        ops.add_res(&a_dev, &b_dev, &out).unwrap();
        let v = out.to_vec().unwrap();
        assert_eq!(v[5 * 8 + 3], a[[5, 3]] + 2.0, "{device:?}");
    }
}

#[test]
fn mandelbrot_scalar_and_short_output() {
    for device in devices() {
        let mb = Mandelbrot::load(&device).unwrap();
        let n = 64usize;
        let q: Vec<num_complex::Complex<f32>> = (0..n * n)
            .map(|k| {
                let (i, j) = (k / n, k % n);
                num_complex::Complex::new(
                    -2.0 + 3.0 * j as f32 / n as f32,
                    -1.5 + 3.0 * i as f32 / n as f32,
                )
            })
            .collect();
        let q = Array1::from_vec(q).into_dyn();
        let iters = mb.mandelbrot(&q.view(), 50).unwrap();
        assert_eq!(iters.shape(), &[n * n]);
        let max = iters.iter().copied().max().unwrap();
        let min = iters.iter().copied().min().unwrap();
        assert!(
            max > min,
            "{device:?}: iterations vary across the plane ({min}..{max})"
        );
    }
}

#[test]
fn gradient_two_out_like_outputs() {
    for device in devices() {
        let g = Gradient::load(&device).unwrap();
        let img = Array2::from_shape_fn((32, 24), |(i, j)| (i as f32) * 0.5 + (j as f32) * 0.25)
            .into_dyn();
        let (grad, theta) = g.gradient(&img.view(), 1).unwrap();
        assert_eq!(grad.shape(), img.shape(), "{device:?}");
        assert_eq!(theta.shape(), img.shape(), "{device:?}");
        assert!(
            grad.iter().any(|&v| v != 0.0),
            "{device:?}: gradient is not all zero"
        );
    }
}

#[test]
fn hsi_roundtrip() {
    for device in devices() {
        let hsi = Hsi::load(&device).unwrap();
        let rgb = Array3::from_shape_fn((8, 8, 3), |(i, j, c)| {
            ((i * 8 + j + c * 7) % 11) as f32 / 10.0
        })
        .into_dyn();
        let out = hsi.rgb2hsi3(&rgb.view()).unwrap();
        assert_eq!(out.shape(), rgb.shape(), "{device:?}");
        let back = hsi.hsi2rgb3(&out.view()).unwrap();
        for (a, b) in rgb.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-3, "{device:?}: {a} vs {b}");
        }
    }
}
