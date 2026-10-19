//! End-to-end: the `demo` interface (c = a + b) on every backend compiled in
//! and present on this machine. Backends without a device are skipped.

use std::path::PathBuf;
use yapocis_rt::{
    available_backends, Backend, CallArg, CallOptions, Device, HostView, HostViewMut, Manifest,
    OutputValue,
};

fn bundle_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../yapocis/bundle")
}

fn open(backend: Backend) -> Option<Device> {
    match Device::open(backend, None) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("skipping {backend:?}: {e}");
            None
        }
    }
}

fn inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.25).collect();
    (a, b)
}

fn run_demo(backend: Backend) {
    let Some(device) = open(backend) else { return };
    let manifest = Manifest::from_path(bundle_dir().join("manifest.json")).unwrap();
    let entry = manifest.interface("demo").unwrap();
    let program = device.load_from_bundle(bundle_dir(), entry).unwrap();
    println!("{backend:?}: {}", device.name());

    let n = 100_000usize;
    let shape = [n];
    let (a, b) = inputs(n);
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    // sum(in a, in b, out_like a): the output is allocated and returned
    let mut args = [
        CallArg::In(HostView::of(&a, &shape)),
        CallArg::In(HostView::of(&b, &shape)),
    ];
    let out = program
        .call("sum", &mut args, &CallOptions::default())
        .unwrap();
    assert_eq!(out.outputs.len(), 1);
    let (oshape, dtype, bytes) = out.into_host().expect("host output");
    assert_eq!(oshape, shape);
    assert_eq!(dtype, yapocis_rt::DType::Float32);
    let c: &[f32] = bytemuck::cast_slice(&bytes);
    assert_eq!(c, &expected[..], "{backend:?} sum");

    // sum_res(in a, in b, out c): written back into the caller's array
    let mut c2 = vec![0f32; n];
    let mut args = [
        CallArg::In(HostView::of(&a, &shape)),
        CallArg::In(HostView::of(&b, &shape)),
        CallArg::InOut(HostViewMut::of(&mut c2, &shape)),
    ];
    let out = program
        .call("sum_res", &mut args, &CallOptions::default())
        .unwrap();
    assert!(matches!(out.outputs[0].value, OutputValue::Written));
    assert_eq!(c2, expected, "{backend:?} sum_res");

    // resident buffers: upload once, call twice, read once
    let a_dev = device.upload(&a, &shape).unwrap();
    let b_dev = device.upload(&b, &shape).unwrap();
    let c_dev = device.alloc::<f32>(&shape).unwrap();
    let mut args = [
        CallArg::Buffer(a_dev.as_any()),
        CallArg::Buffer(b_dev.as_any()),
        CallArg::Buffer(c_dev.as_any()),
    ];
    let opts = CallOptions {
        async_exec: true,
        ..Default::default()
    };
    let first = program.call("sum_res", &mut args, &opts).unwrap();
    assert!(first.event.is_some());
    let mut args = [
        CallArg::Buffer(c_dev.as_any()),
        CallArg::Buffer(b_dev.as_any()),
        CallArg::Buffer(c_dev.as_any()),
    ];
    program.call("sum_res", &mut args, &opts).unwrap();
    device.wait_all().unwrap();
    let twice = c_dev.to_vec().unwrap();
    let expected2: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y + y).collect();
    assert_eq!(twice, expected2, "{backend:?} resident chain");

    // async out_like comes back as a device buffer
    let mut args = [
        CallArg::Buffer(a_dev.as_any()),
        CallArg::Buffer(b_dev.as_any()),
    ];
    let out = program.call("sum", &mut args, &opts).unwrap();
    out.event.as_ref().unwrap().wait().unwrap();
    match &out.outputs[0].value {
        OutputValue::Device(buf) => {
            let v = buf.clone().typed::<f32>().unwrap().to_vec().unwrap();
            assert_eq!(v, expected, "{backend:?} async out_like");
        }
        other => panic!("expected a device buffer, got {other:?}"),
    }

    // explicit global size smaller than the array only touches a prefix
    let mut c3 = vec![-1f32; n];
    let mut args = [
        CallArg::In(HostView::of(&a, &shape)),
        CallArg::In(HostView::of(&b, &shape)),
        CallArg::InOut(HostViewMut::of(&mut c3, &shape)),
    ];
    let opts = CallOptions {
        global: Some(vec![10]),
        ..Default::default()
    };
    program.call("sum_res", &mut args, &opts).unwrap();
    assert_eq!(&c3[..10], &expected[..10]);
    assert!(
        c3[10..].iter().all(|&v| v == -1.0),
        "{backend:?} partial launch"
    );
}

#[test]
fn demo_on_every_available_backend() {
    let backends = available_backends();
    assert!(
        !backends.is_empty(),
        "build with --features ocl,metal and/or cuda"
    );
    for backend in backends {
        run_demo(backend);
    }
}

#[test]
fn dtype_mismatch_is_an_error() {
    let Some(device) = available_backends().into_iter().find_map(open) else {
        return;
    };
    let manifest = Manifest::from_path(bundle_dir().join("manifest.json")).unwrap();
    let program = device
        .load_from_bundle(bundle_dir(), manifest.interface("demo").unwrap())
        .unwrap();
    let a = vec![1i32; 4];
    let b = vec![1f32; 4];
    let mut args = [
        CallArg::In(HostView::of(&a, &[4])),
        CallArg::In(HostView::of(&b, &[4])),
    ];
    let err = program
        .call("sum", &mut args, &CallOptions::default())
        .unwrap_err();
    assert!(matches!(err, yapocis_rt::Error::DType { .. }), "{err}");
    let mut args = [CallArg::In(HostView::of(&b, &[4]))];
    let err = program
        .call("sum", &mut args, &CallOptions::default())
        .unwrap_err();
    assert!(matches!(err, yapocis_rt::Error::Derive(_)), "{err}");
}

/// Host arrays are bound to cached device buffers across calls (as in the
/// Python device managers); contents are uploaded every call, so in-place
/// changes to a bound array are seen, and outputs reuse free buffers.
fn run_buffer_cache(backend: Backend) {
    let Some(device) = open(backend) else { return };
    let manifest = Manifest::from_path(bundle_dir().join("manifest.json")).unwrap();
    let program = device
        .load_from_bundle(bundle_dir(), manifest.interface("demo").unwrap())
        .unwrap();
    let n = 4096usize;
    let shape = [n];
    let (mut a, b) = inputs(n);
    let sum = |a: &[f32], b: &[f32]| -> Vec<f32> {
        let mut args = [
            CallArg::In(HostView::of(a, &shape)),
            CallArg::In(HostView::of(b, &shape)),
        ];
        let out = program
            .call("sum", &mut args, &CallOptions::default())
            .unwrap();
        let (_, _, data) = out.into_host().unwrap();
        data.into_vec::<f32>().expect("f32 output")
    };
    let add = |a: &[f32], b: &[f32]| -> Vec<f32> { a.iter().zip(b).map(|(x, y)| x + y).collect() };

    for _ in 0..3 {
        assert_eq!(sum(&a, &b), add(&a, &b), "{backend:?} repeated sum");
    }
    let s = device.buffer_cache_stats();
    // the first call allocates a, b and the output; later calls rebind a and
    // b and reuse the output buffer
    assert_eq!(
        (s.misses, s.bound_hits, s.size_hits),
        (3, 4, 2),
        "{backend:?} {s:?}"
    );
    assert_eq!(s.entries, 3);

    // same address, new contents: must be re-uploaded
    for v in a.iter_mut() {
        *v += 1000.0;
    }
    assert_eq!(sum(&a, &b), add(&a, &b), "{backend:?} in-place change");

    // the same host array in two arguments gets two distinct buffers
    assert_eq!(sum(&a, &a), add(&a, &a), "{backend:?} aliased arguments");

    // inout is written back correctly when its buffer is reused
    let mut c = vec![0f32; n];
    for k in 0..2 {
        let shifted: Vec<f32> = a.iter().map(|x| x + k as f32).collect();
        let mut args = [
            CallArg::In(HostView::of(&shifted, &shape)),
            CallArg::In(HostView::of(&b, &shape)),
            CallArg::InOut(HostViewMut::of(&mut c, &shape)),
        ];
        program
            .call("sum_res", &mut args, &CallOptions::default())
            .unwrap();
        assert_eq!(c, add(&shifted, &b), "{backend:?} sum_res reuse {k}");
    }

    // a different length does not take a cached buffer of another size
    let small = [64usize];
    let mut args = [
        CallArg::In(HostView::of(&a[..64], &small)),
        CallArg::In(HostView::of(&b[..64], &small)),
    ];
    let out = program
        .call("sum", &mut args, &CallOptions::default())
        .unwrap();
    let (oshape, _, data) = out.into_host().unwrap();
    assert_eq!(oshape, small);
    assert_eq!(data.into_vec::<f32>().unwrap(), add(&a[..64], &b[..64]));

    device.purge_buffers();
    assert_eq!(device.buffer_cache_stats().entries, 0);
    assert_eq!(sum(&a, &b), add(&a, &b), "{backend:?} after purge");
}

#[test]
fn buffer_cache_on_every_available_backend() {
    for backend in available_backends() {
        run_buffer_cache(backend);
    }
}

/// `resident` parameters are never uploaded: they take device buffers, and a
/// host array passed to one is an error instead of a silent write.
fn run_resident(backend: Backend) {
    let Some(device) = open(backend) else { return };
    let manifest = Manifest::from_path(bundle_dir().join("manifest.json")).unwrap();
    let program = device
        .load_from_bundle(bundle_dir(), manifest.interface("colormath").unwrap())
        .unwrap();
    let n = 1024usize;
    let shape = [n, 3];
    let mut xyz: Vec<f32> = (0..n * 3).map(|i| 0.1 + (i % 97) as f32 / 100.0).collect();
    let opts = CallOptions {
        global: Some(vec![n]),
        ..Default::default()
    };

    // reference: the `in`/`out_like` kernel
    let mut args = [CallArg::In(HostView::of(&xyz, &shape))];
    let (_, _, data) = program
        .call("xyz2xyY", &mut args, &opts)
        .unwrap()
        .into_host()
        .unwrap();
    let expected = data.into_vec::<f32>().unwrap();

    // a host array for a resident parameter is rejected
    let mut out_host = vec![0f32; n * 3];
    let mut args = [
        CallArg::In(HostView::of(&xyz, &shape)),
        CallArg::InOut(HostViewMut::of(&mut out_host, &shape)),
    ];
    let err = program.call("xyz2xyY_res", &mut args, &opts).unwrap_err();
    assert!(
        matches!(&err, yapocis_rt::Error::ArgKind { role, .. } if role == "resident"),
        "{backend:?}: {err}"
    );

    // device buffers: uploaded once by the caller, then used in place
    let input = device.upload(&xyz, &shape).unwrap();
    let output = device.alloc::<f32>(&shape).unwrap();
    let before = device.buffer_cache_stats();
    let mut args = [
        CallArg::Buffer(input.as_any()),
        CallArg::Buffer(output.as_any()),
    ];
    let out = program.call("xyz2xyY_res", &mut args, &opts).unwrap();
    assert!(
        out.outputs.is_empty(),
        "resident results stay on the device"
    );
    assert_eq!(
        output.to_vec().unwrap(),
        expected,
        "{backend:?} resident result"
    );

    // changing the host array afterwards changes nothing on the device: the
    // call reads the buffer, it does not write the host data again
    for v in xyz.iter_mut() {
        *v += 0.5;
    }
    let mut args = [
        CallArg::Buffer(input.as_any()),
        CallArg::Buffer(output.as_any()),
    ];
    program.call("xyz2xyY_res", &mut args, &opts).unwrap();
    assert_eq!(
        output.to_vec().unwrap(),
        expected,
        "{backend:?} no re-upload"
    );
    assert_eq!(
        device.buffer_cache_stats(),
        before,
        "{backend:?}: resident calls take no temporary buffers"
    );
}

#[test]
fn resident_parameters_are_never_uploaded() {
    for backend in available_backends() {
        run_resident(backend);
    }
}
