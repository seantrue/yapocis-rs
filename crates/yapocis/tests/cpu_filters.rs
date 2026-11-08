//! The CPU ports of anisodiff, gradient, median3x3 and zcs against the GPU
//! backends on this machine, including kernels the parity fixtures do not
//! record (`filterImage3`, `zcs3`). Skips when the CPU backend or no other
//! backend is available.

use std::path::PathBuf;
use yapocis::rt::{CallArg, CallOptions, HostView, OutputValue};
use yapocis::{available_backends, Backend, Device, MANIFEST};

mod common;

const W: usize = 37;
const H: usize = 29;

/// Deterministic values in [-1, 1).
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// One call: `interface.kernel(scalars_before..., image, scalars_after...)`.
struct Case<'a> {
    interface: &'a str,
    kernel: &'a str,
    image: &'a [f32],
    shape: &'a [usize],
    scalars_before: Vec<f64>,
    scalars_after: Vec<f64>,
    global: Option<usize>,
}

/// Run `case` on `device`, returning every host output.
fn run(device: &Device, case: &Case) -> Vec<Vec<u8>> {
    let Case {
        interface,
        kernel,
        image,
        shape,
        ref scalars_before,
        ref scalars_after,
        global,
    } = *case;
    let entry = MANIFEST.interface(interface).unwrap();
    let src = &entry.sources[device.backend().source_backend().name()];
    let text = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("bundle")
            .join(&src.path),
    )
    .unwrap();
    let program = device.load(entry, &text).unwrap();
    let mut args: Vec<CallArg<'_>> = scalars_before.iter().map(|&s| CallArg::Scalar(s)).collect();
    args.push(CallArg::In(HostView::of(image, shape)));
    args.extend(scalars_after.iter().map(|&s| CallArg::Scalar(s)));
    let opts = CallOptions {
        global: global.map(|g| vec![g]),
        ..Default::default()
    };
    let out = program.call(kernel, &mut args, &opts).unwrap();
    out.outputs
        .into_iter()
        .filter_map(|o| match o.value {
            OutputValue::Host { bytes, .. } => Some(bytes.as_bytes().to_vec()),
            _ => None,
        })
        .collect()
}

#[test]
fn cpu_filters_match_gpu_backends() {
    let Ok(cpu) = Device::open(Backend::RustCpu, None) else {
        eprintln!("no CPU backend in this build");
        return;
    };
    let others: Vec<Device> = available_backends()
        .into_iter()
        .filter(|&b| b != Backend::RustCpu)
        .filter_map(|b| Device::open(b, None).ok())
        .collect();
    if others.is_empty() {
        eprintln!("no GPU backend to compare with");
        return;
    }
    let (rtol, atol) = common::tolerances(1e-5, 1e-5);
    let mut failures = Vec::new();
    for depth in 1..=4usize {
        // filterImage3 reads float3, 16 bytes a pixel: give it 4 floats a pixel.
        let stride = if depth == 3 { 4 } else { depth };
        let shape = [W, H, stride];
        let image = values(W * H * stride, depth as u64);
        let filter = [
            "filterImage",
            "filterImage2",
            "filterImage3",
            "filterImage4",
        ][depth - 1];
        let zcs = format!("zcs{depth}");
        let zshape = [W, H, depth];
        let zimage = values(W * H * depth, 10 + depth as u64);
        let cases = [
            Case {
                interface: "anisodiff",
                kernel: filter,
                image: &image,
                shape: &shape,
                scalars_before: vec![],
                scalars_after: vec![0.3, 0.2],
                global: Some(W * H),
            },
            Case {
                interface: "zcs",
                kernel: &zcs,
                image: &zimage,
                shape: &zshape,
                scalars_before: vec![0.05],
                scalars_after: vec![],
                global: Some(W * H),
            },
        ];
        for case in &cases {
            let (iface, kernel) = (case.interface, case.kernel);
            let want_cpu = run(&cpu, case);
            for dev in &others {
                let got = run(dev, case);
                for (c, g) in want_cpu.iter().zip(&got) {
                    let (c, g): (&[f32], &[f32]) =
                        (bytemuck::cast_slice(c), bytemuck::cast_slice(g));
                    // The fourth float of a float3 pixel is padding: compare only real channels.
                    let keep = |v: &[f32]| -> Vec<f32> {
                        v.iter()
                            .enumerate()
                            .filter(|(i, _)| !(depth == 3 && iface == "anisodiff" && i % 4 == 3))
                            .map(|(_, &x)| x)
                            .collect()
                    };
                    if let Err(e) = common::allclose_f32(&keep(c), &keep(g), rtol, atol) {
                        failures.push(format!("{iface}.{kernel} vs {dev:?}: {e}"));
                    }
                }
            }
        }
    }
    let image = values(W * H, 99);
    for (iface, kernel, after) in [
        ("median3x3", "median3x3", vec![]),
        ("gradient", "gradient", vec![2.0]),
    ] {
        let case = Case {
            interface: iface,
            kernel,
            image: &image,
            shape: &[W, H],
            scalars_before: vec![],
            scalars_after: after,
            global: None,
        };
        let want_cpu = run(&cpu, &case);
        for dev in &others {
            let got = run(dev, &case);
            for (c, g) in want_cpu.iter().zip(&got) {
                if let Err(e) = common::allclose_f32(
                    bytemuck::cast_slice(c),
                    bytemuck::cast_slice(g),
                    rtol,
                    atol,
                ) {
                    failures.push(format!("{iface}.{kernel} vs {dev:?}: {e}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
