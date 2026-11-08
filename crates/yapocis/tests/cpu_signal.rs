//! The CPU ports of `fft.cl`, `simplex.cl` and `truetrace.cl` against the
//! OpenCL (and Metal) kernels on this machine, for inputs the parity fixtures
//! do not cover: radix-2 and DFT FFT paths, batches, every window, large
//! simplex domains, and jump flood plus marching squares on a bigger mask.
//! Skips unless the CPU backend and at least one GPU backend are built in and
//! open a device.

use ndarray::{Array1, Array2};
use num_complex::Complex;
use yapocis::lib::fft::Window;
use yapocis::lib::simplex::{Params2d, Params3d};
use yapocis::lib::{Fft, Simplex, TrueTrace};
use yapocis::{available_backends, Backend, Device};

fn open(backend: Backend) -> Option<Device> {
    if !available_backends().contains(&backend) {
        return None;
    }
    Device::open(backend, None).ok()
}

/// The CPU device and the GPU devices to compare it with.
fn pairs() -> Option<(Device, Vec<Device>)> {
    let cpu = open(Backend::RustCpu)?;
    let gpus: Vec<Device> = [Backend::Ocl, Backend::Metal]
        .into_iter()
        .filter_map(open)
        .collect();
    if gpus.is_empty() {
        eprintln!("no GPU backend to compare the CPU backend with");
        return None;
    }
    Some((cpu, gpus))
}

/// Largest `|a - b| - (atol + rtol * |b|)`, or `None` when within tolerance.
fn worst(a: &[f32], b: &[f32], rtol: f32, atol: f32) -> Option<(usize, f32, f32)> {
    assert_eq!(a.len(), b.len());
    let mut out = None;
    let mut worst = 0f32;
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let excess = (x - y).abs() - (atol + rtol * y.abs());
        if excess > worst {
            worst = excess;
            out = Some((i, x, y));
        }
    }
    out
}

fn assert_close(label: &str, cpu: &[f32], gpu: &[f32], rtol: f32, atol: f32) {
    if let Some((i, x, y)) = worst(cpu, gpu, rtol, atol) {
        let bad = cpu
            .iter()
            .zip(gpu)
            .filter(|(x, y)| (*x - *y).abs() > atol + rtol * y.abs())
            .count();
        panic!(
            "{label}: {bad} of {} differ; worst element {i}: cpu {x}, gpu {y}",
            cpu.len()
        );
    }
}

fn flat(z: &[Complex<f32>]) -> Vec<f32> {
    z.iter().flat_map(|c| [c.re, c.im]).collect()
}

fn signal(n: usize) -> Array1<Complex<f32>> {
    Array1::from_shape_fn(n, |i| {
        let t = i as f32 / n as f32;
        Complex::new(
            (2.0 * std::f32::consts::PI * 5.0 * t).sin() + 0.3 * (37.0 * t).cos(),
            0.25 * (11.0 * t).sin(),
        )
    })
}

#[test]
fn fft_paths_match_gpu() {
    let Some((cpu, gpus)) = pairs() else { return };
    let c = Fft::load(&cpu).unwrap();
    for gpu in &gpus {
        let g = Fft::load(gpu).unwrap();
        let label = format!("{:?}", gpu.backend());
        // 64 and 256 take the radix-2 path; 512 and 1000 the DFT (512 runs
        // without a local size, so the radix-2 path is never taken). The DFT's
        // angles reach thousands of radians, where Metal's fast cos/sin drift,
        // so only OpenCL is held to it.
        let sizes: &[usize] = if gpu.backend() == Backend::Ocl {
            &[64, 256, 512, 1000]
        } else {
            &[64, 256]
        };
        for &n in sizes {
            let x = signal(n);
            let tol = if n > 256 { 2e-3 } else { 1e-4 };
            let fc = c.forward(&x.view()).unwrap();
            let fg = g.forward(&x.view()).unwrap();
            assert_close(
                &format!("{label} forward {n}"),
                &flat(fc.as_slice().unwrap()),
                &flat(fg.as_slice().unwrap()),
                tol,
                tol,
            );
            let ic = c.inverse(&fc.view()).unwrap();
            let ig = g.inverse(&fg.view()).unwrap();
            assert_close(
                &format!("{label} inverse {n}"),
                &flat(ic.as_slice().unwrap()),
                &flat(ig.as_slice().unwrap()),
                tol,
                tol,
            );
        }
        let batch = Array2::from_shape_fn((6, 128), |(b, i)| {
            Complex::new(
                ((b + 1) as f32 * i as f32 * 0.07).sin(),
                (i as f32 * 0.01).cos(),
            )
        });
        let bc = c.batch_forward(&batch.view()).unwrap();
        let bg = g.batch_forward(&batch.view()).unwrap();
        assert_close(
            &format!("{label} batch"),
            &flat(bc.as_slice().unwrap()),
            &flat(bg.as_slice().unwrap()),
            1e-4,
            1e-4,
        );
        // `window` takes n from the launch (`local size * groups`). For a length
        // that is not a multiple of the group size, Apple's OpenCL rounds the
        // launch up (300 runs as 512); the port takes the global size, as
        // PoCL does. A power of two avoids the difference.
        let real: Array1<f32> = Array1::from_shape_fn(256, |i| (i as f32 * 0.1).sin() + 1.0);
        for w in [
            Window::Rectangular,
            Window::Hann,
            Window::Hamming,
            Window::Blackman,
        ] {
            let wc = c.apply_window(&real.view(), w).unwrap();
            let wg = g.apply_window(&real.view(), w).unwrap();
            assert_close(
                &format!("{label} window {w:?}"),
                wc.as_slice().unwrap(),
                wg.as_slice().unwrap(),
                1e-5,
                1e-6,
            );
        }
        let pc = c.power_spectrum(&fc_of(&c, 256).view()).unwrap();
        let pg = g.power_spectrum(&fc_of(&c, 256).view()).unwrap();
        assert_close(
            &format!("{label} power"),
            pc.as_slice().unwrap(),
            pg.as_slice().unwrap(),
            1e-6,
            0.0,
        );
    }
}

fn fc_of(fft: &Fft, n: usize) -> Array1<Complex<f32>> {
    fft.forward(&signal(n).view()).unwrap()
}

#[test]
fn simplex_matches_gpu() {
    let Some((cpu, gpus)) = pairs() else { return };
    let c = Simplex::load(&cpu).unwrap();
    for gpu in &gpus {
        let g = Simplex::load(gpu).unwrap();
        let label = format!("{:?}", gpu.backend());
        for (tileable, scale) in [(false, 3.0f32), (true, 2.0)] {
            let p = Params2d {
                seed: 1234,
                scale,
                offset: (-7.25, 3.5),
                octaves: 5,
                tileable,
                ..Default::default()
            };
            let nc = c.simplex2d(200, 150, &p).unwrap();
            let ng = g.simplex2d(200, 150, &p).unwrap();
            assert_close(
                &format!("{label} simplex2d tileable={tileable}"),
                nc.as_slice().unwrap(),
                ng.as_slice().unwrap(),
                1e-4,
                1e-4,
            );
        }
        // simplex3d: the noise is discontinuous at simplex boundaries (radius
        // 0.6), so where the GPU's fast math and the exact port fall on
        // opposite sides a pixel jumps by up to ~1e-3. Hold most pixels to
        // 1e-4 and all to 2e-3. Only OpenCL: Metal's 3-D noise is a different
        // field (its recorded fixtures differ from OpenCL's by ~0.5).
        if gpu.backend() == Backend::Ocl {
            let p = Params3d {
                seed: 99,
                scale: 6.0,
                offset: (-3.1, 2.2, 0.7),
                octaves: 3,
                ..Default::default()
            };
            let nc = c.simplex3d(40, 30, 10, &p).unwrap();
            let ng = g.simplex3d(40, 30, 10, &p).unwrap();
            let (nc, ng) = (nc.as_slice().unwrap(), ng.as_slice().unwrap());
            assert_close(&format!("{label} simplex3d"), nc, ng, 0.0, 2e-3);
            let off = nc
                .iter()
                .zip(ng)
                .filter(|(x, y)| (*x - *y).abs() > 1e-4)
                .count();
            assert!(
                off * 50 < nc.len(),
                "{label} simplex3d: {off} of {} pixels differ by more than 1e-4",
                nc.len()
            );
        }
    }
}

#[test]
fn truetrace_segments_match_gpu() {
    let Some((cpu, gpus)) = pairs() else { return };
    let (h, w) = (90usize, 120usize);
    let mask = Array2::from_shape_fn((h, w), |(y, x)| {
        let (fx, fy) = (x as f32 - 60.0, y as f32 - 45.0);
        let ring = (fx * fx + fy * fy).sqrt();
        ((20.0..34.0).contains(&ring) || (x / 9 + y / 7) % 5 == 0) as u8
    });
    let c = TrueTrace::load(&cpu).unwrap();
    let (sc, size_c) = c.trace_segments(&mask.view(), 0.0).unwrap();
    for gpu in &gpus {
        let g = TrueTrace::load(gpu).unwrap();
        let (sg, size_g) = g.trace_segments(&mask.view(), 0.0).unwrap();
        let label = format!("{:?}", gpu.backend());
        assert_eq!(size_c, size_g);
        assert_eq!(sc.nrows(), sg.nrows(), "{label}: segment count");
        // Both are sorted by `trace_segments`, so rows line up.
        assert_close(
            &format!("{label} segments"),
            sc.as_slice().unwrap(),
            sg.as_slice().unwrap(),
            1e-4,
            1e-4,
        );
    }
}
