//! The CPU ports of the `vgg16` kernels against the GPU backends on this
//! machine. No Python fixture covers the `vgg16` interface (its layers are
//! checked only through full-model runs that need the weights), so each
//! kernel runs on the same inputs on the CPU and on every other device, and
//! the results must agree to the library parity tolerances. Skips when the
//! CPU backend or any other backend is missing.

mod common;

use yapocis::rt::{CallArg, CallOptions, HostView, HostViewMut, OutputValue, Program};
use yapocis::{Backend, Device, MANIFEST};

/// A deterministic array in `[-0.5, 0.5)` (or `[0.1, 1.1)` when `positive`).
struct Array {
    shape: Vec<usize>,
    data: Vec<f32>,
}

fn array(shape: &[usize], seed: usize, positive: bool) -> Array {
    let n: usize = shape.iter().product();
    let data = (0..n)
        .map(|i| {
            let v = ((i * 7919 + seed * 104729) % 1000) as f32 / 1000.0;
            if positive {
                v + 0.1
            } else {
                v - 0.5
            }
        })
        .collect();
    Array {
        shape: shape.to_vec(),
        data,
    }
}

/// One kernel call: inputs, the `out` arrays (zeroed) and scalars, in user-argument order.
enum Arg {
    In(Array),
    Out(Vec<usize>),
    Scalar(f64),
}

fn cases() -> Vec<(&'static str, Vec<Arg>)> {
    use Arg::*;
    let (c, h, w, oc) = (3, 8, 10, 4);
    let conv = || {
        vec![
            In(array(&[c, h, w], 1, false)),
            In(array(&[oc, c, 3, 3], 2, false)),
            In(array(&[oc], 3, false)),
            Out(vec![oc, h, w]),
            Scalar(oc as f64),
        ]
    };
    vec![
        ("conv2d_3x3", conv()),
        ("conv2d_3x3_optimized", conv()),
        ("relu_activation", vec![In(array(&[37], 4, false))]),
        ("softmax", vec![In(array(&[37], 5, false))]),
        (
            "max_pool_2x2",
            vec![In(array(&[c, h, w], 6, false)), Out(vec![c, h / 2, w / 2])],
        ),
        (
            // The dispatch size is (height_of, depth_of, width_of), and in a
            // dispatch size `depth_of` is shape[0], not shape[2] as in the
            // arguments: w runs to C. Its output is out_like, so elements no
            // work item writes are whatever the buffer held; C == W writes all.
            "batch_norm",
            vec![
                In(array(&[c, h, c], 7, false)),
                In(array(&[c], 8, false)),
                In(array(&[c], 9, false)),
                In(array(&[c], 10, false)),
                In(array(&[c], 11, true)),
                Scalar(1e-5),
            ],
        ),
        (
            "fully_connected",
            vec![
                In(array(&[12], 12, false)),
                In(array(&[5 * 12], 13, false)),
                In(array(&[5], 14, false)),
                Out(vec![5]),
            ],
        ),
        (
            // M = height_of A, N = height_of B, K = width_of A
            "matrix_multiply_simd",
            vec![
                In(array(&[3, 4], 15, false)),
                In(array(&[3, 5], 16, false)),
                In(array(&[5], 17, false)),
                Out(vec![4, 5]),
            ],
        ),
    ]
}

/// Every output of `kernel` on `program`, as host vectors.
fn run(program: &Program, kernel: &str, case: &[Arg]) -> Vec<Vec<f32>> {
    let mut outs: Vec<Vec<f32>> = case
        .iter()
        .filter_map(|a| match a {
            Arg::Out(shape) => Some(vec![0f32; shape.iter().product()]),
            _ => None,
        })
        .collect();
    let mut out_slots = outs.iter_mut();
    let mut args: Vec<CallArg<'_>> = case
        .iter()
        .map(|a| match a {
            Arg::In(x) => CallArg::In(HostView::of(&x.data, &x.shape)),
            Arg::Out(shape) => CallArg::InOut(HostViewMut::of(out_slots.next().unwrap(), shape)),
            Arg::Scalar(s) => CallArg::Scalar(*s),
        })
        .collect();
    let result = program
        .call(kernel, &mut args, &CallOptions::default())
        .unwrap_or_else(|e| panic!("{kernel}: {e}"));
    drop(args);
    let mut written = outs.into_iter();
    result
        .outputs
        .into_iter()
        .map(|o| match o.value {
            OutputValue::Written => written.next().unwrap(),
            OutputValue::Host { bytes, .. } => bytemuck::cast_slice(bytes.as_bytes()).to_vec(),
            OutputValue::Device(_) => panic!("{kernel}: unexpected device output"),
        })
        .collect()
}

fn load(device: &Device) -> Program {
    let entry = MANIFEST.interface("vgg16").unwrap();
    let source =
        yapocis::interfaces::vgg16::source(device.backend().source_backend().name()).unwrap();
    device.load(entry, source).unwrap()
}

#[test]
fn vgg16_cpu_matches_gpu_backends() {
    let devices = common::devices();
    let Some(cpu) = devices.iter().find(|d| d.backend() == Backend::RustCpu) else {
        eprintln!("no CPU backend in this build");
        return;
    };
    let others: Vec<&Device> = devices
        .iter()
        .filter(|d| d.backend() != Backend::RustCpu)
        .collect();
    if others.is_empty() {
        eprintln!("no GPU backend to compare the CPU with");
        return;
    }
    let cpu_program = load(cpu);
    let (rtol, atol) = common::tolerances(1e-5, 1e-5);
    let mut failures = Vec::new();
    for other in others {
        let program = load(other);
        for (kernel, case) in cases() {
            // Metal's vgg16 matrix_multiply_simd is cnn's 1-D form (row = gid % M,
            // col = gid / M); OpenCL's, which the CPU ports, is 2-D.
            if other.backend() == Backend::Metal && kernel == "matrix_multiply_simd" {
                continue;
            }
            let want = run(&program, kernel, &case);
            let got = run(&cpu_program, kernel, &case);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                if let Err(e) = common::allclose_f32(g, w, rtol, atol) {
                    failures.push(format!("{other:?} {kernel} output {i}: {e}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
