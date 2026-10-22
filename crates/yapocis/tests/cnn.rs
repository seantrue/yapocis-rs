//! CNN layer parity (every `cnn` kernel against Python outputs) and the
//! VGG16 / ResNet50 ports (gated on the model weights being installed).

mod common;

use common::{allclose_f32, open_npz, read_any, read_array};
use ndarray::{ArrayD, ArrayView3, IxDyn};
use yapocis::lib::cnn::{preprocess_resnet50, preprocess_vgg16, rgb_from_hwc, top_k, weights_path};
use yapocis::lib::resnet50::ResNet50Layer;
use yapocis::lib::vgg16::Vgg16Layer;
use yapocis::lib::{ResNet50, Vgg16};
use yapocis::rt::{CallArg, CallOptions, HostView, HostViewMut, OutputValue};
use yapocis::{Device, MANIFEST};

#[test]
fn cnn_kernels_match_python() {
    let mut checked = 0;
    for device in common::devices() {
        let backend = device.backend();
        let Some(index) = common::lib_index(backend) else {
            eprintln!("no library parity fixtures for {backend:?}");
            continue;
        };
        let Some(cases) = index["generators"].get("cnn").and_then(|c| c.as_array()) else {
            eprintln!("no cnn cases for {backend:?}");
            continue;
        };
        let rtol = index["rtol"].as_f64().unwrap();
        let atol = index["atol"].as_f64().unwrap();
        let entry = MANIFEST.interface("cnn").unwrap();
        let program = device
            .load(
                entry,
                yapocis::interfaces::cnn::source(backend.name()).unwrap(),
            )
            .unwrap();
        let dir = common::parity_lib_dir(backend);
        let mut failures = Vec::new();
        for case in cases {
            let kernel = case["kernel"].as_str().unwrap();
            let mut npz = open_npz(&dir.join(case["file"].as_str().unwrap()));
            // Inputs and zeroed `out` arrays
            let mut inputs: Vec<Option<common::Npy>> = Vec::new();
            let mut outs: Vec<Option<(Vec<usize>, Vec<f32>)>> = Vec::new();
            for arg in case["args"].as_array().unwrap() {
                if let Some(key) = arg.get("array").and_then(|k| k.as_str()) {
                    inputs.push(Some(read_any(&mut npz, key)));
                    outs.push(None);
                } else if let Some(shape) = arg.get("shape") {
                    let shape: Vec<usize> = serde_json::from_value(shape.clone()).unwrap();
                    let n: usize = shape.iter().product();
                    inputs.push(None);
                    outs.push(Some((shape, vec![0f32; n])));
                } else {
                    inputs.push(None);
                    outs.push(None);
                }
            }
            let expected: Vec<ArrayD<f32>> = case["outputs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|k| read_array::<f32>(&mut npz, k.as_str().unwrap()))
                .collect();
            let args_json = case["args"].as_array().unwrap();
            let mut args: Vec<CallArg<'_>> = Vec::new();
            for ((arg, input), out) in args_json.iter().zip(&inputs).zip(outs.iter_mut()) {
                if let Some(n) = input {
                    args.push(CallArg::In(HostView {
                        dtype: n.dtype,
                        shape: &n.shape,
                        bytes: &n.bytes,
                    }));
                } else if let Some((shape, data)) = out {
                    args.push(CallArg::InOut(HostViewMut::of(data.as_mut_slice(), shape)));
                } else {
                    args.push(CallArg::Scalar(arg["scalar"].as_f64().unwrap()));
                }
            }
            let out = match program.call(kernel, &mut args, &CallOptions::default()) {
                Ok(o) => o,
                Err(e) => {
                    failures.push(format!("{kernel}: {e}"));
                    continue;
                }
            };
            drop(args);
            // Returned values in order: `out` params (written in place) then out_like host arrays
            let mut got: Vec<Vec<f32>> = Vec::new();
            let mut out_iter = outs.iter().filter_map(|o| o.as_ref());
            for o in &out.outputs {
                match &o.value {
                    OutputValue::Written => got.push(out_iter.next().expect("out param").1.clone()),
                    OutputValue::Host { bytes, .. } => {
                        got.push(bytemuck::cast_slice(bytes).to_vec())
                    }
                    OutputValue::Device(_) => panic!("unexpected device output"),
                }
            }
            if got.len() != expected.len() {
                failures.push(format!(
                    "{kernel}: {} outputs, expected {}",
                    got.len(),
                    expected.len()
                ));
                continue;
            }
            for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
                if let Err(msg) = common::allclose_f32(
                    g,
                    e.as_slice().unwrap(),
                    common::tolerances(rtol, atol).0,
                    common::tolerances(rtol, atol).1,
                ) {
                    failures.push(format!("{kernel}: output {i}: {msg}"));
                }
            }
            checked += 1;
        }
        assert!(failures.is_empty(), "{backend:?}: {}", failures.join("\n"));
        println!("{backend:?}: {} cnn kernels match", cases.len());
    }
    assert!(checked > 0, "no cnn parity cases were checked");
}

#[test]
fn labels_and_top_k() {
    assert_eq!(Vgg16::labels().len(), 1000);
    assert_eq!(ResNet50::labels().len(), 1000);
    assert_eq!(ResNet50::labels()[207], "golden retriever");
    let top = top_k(&[0.1, 0.5, 0.2, 0.5], 3);
    assert_eq!(top, vec![(1, 0.5), (3, 0.5), (2, 0.2)]);
}

#[test]
fn preprocessing_shapes() {
    let img = image::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/golden_retriever.jpg"
    ))
    .unwrap()
    .to_rgb8();
    let v = preprocess_vgg16(&img);
    assert_eq!(v.dim(), (3, 224, 224));
    let r = preprocess_resnet50(&img);
    assert_eq!(r.dim(), (3, 224, 224));
    // normalised values stay within the ImageNet range
    assert!(r.iter().all(|&x| (-2.2..=2.7).contains(&x)));
}

/// Full-model checks against the Python outputs, where the weights are installed
/// (`YAPOCIS_MODELS`) and the fixture was generated there.
fn model_case(device: &Device, model: &str) -> bool {
    let backend = device.backend();
    let Some(index) = common::lib_index(backend) else {
        return false;
    };
    let Some(cases) = index["generators"].get("models").and_then(|c| c.as_array()) else {
        return false;
    };
    let Some(case) = cases.iter().find(|c| c["model"] == model) else {
        eprintln!("no {model} model fixture for {backend:?}");
        return false;
    };
    if weights_path(model).is_err() {
        eprintln!("skipping {model}: weights not installed");
        return false;
    }
    let mut npz = open_npz(&common::parity_lib_dir(backend).join(case["file"].as_str().unwrap()));
    let image_hwc: ArrayD<u8> = read_array(&mut npz, "image");
    let pre: ArrayD<f32> = read_array(&mut npz, "preprocessed");
    let probs: ArrayD<f32> = read_array(&mut npz, "probs");
    let embedding: ArrayD<f32> = read_array(&mut npz, "embedding");
    let top5: ArrayD<i64> = read_array(&mut npz, "top5_indices");
    let pre3: ArrayView3<'_, f32> = pre.view().into_dimensionality().unwrap();
    let img = rgb_from_hwc(&image_hwc.view().into_dimensionality().unwrap());

    let (got_probs, got_emb, got_pre, got_top) = match model {
        "resnet50" => {
            let net = ResNet50::load_default(device).unwrap();
            (
                net.forward(&pre3, ResNet50Layer::Probabilities).unwrap(),
                net.forward(&pre3, ResNet50Layer::Pool).unwrap(),
                net.preprocess(&img),
                net.predict(&img, 5).unwrap(),
            )
        }
        _ => {
            let net = Vgg16::load_default(device).unwrap();
            (
                net.forward(&pre3, Vgg16Layer::Probabilities).unwrap(),
                net.forward(&pre3, Vgg16Layer::Fc6).unwrap(),
                net.preprocess(&img),
                net.predict(&img, 5).unwrap(),
            )
        }
    };
    // network from the stored preprocessed tensor: tight
    allclose_f32(
        got_probs.as_slice().unwrap(),
        probs.as_slice().unwrap(),
        1e-3,
        1e-5,
    )
    .unwrap_or_else(|e| panic!("{backend:?} {model} probabilities: {e}"));
    allclose_f32(
        got_emb.as_slice().unwrap(),
        embedding.as_slice().unwrap(),
        1e-3,
        1e-3,
    )
    .unwrap_or_else(|e| panic!("{backend:?} {model} embedding: {e}"));
    // preprocessing: resampling differs by up to one 8-bit step on a few pixels
    let diff: f32 = got_pre
        .iter()
        .zip(pre.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / pre.len() as f32;
    assert!(
        diff < 0.02,
        "{backend:?} {model} preprocessing mean abs diff {diff}"
    );
    // end to end: same top-1 class
    assert_eq!(
        got_top[0].0 as i64,
        top5[[0]],
        "{backend:?} {model} top-1 ({:?} vs {:?})",
        got_top,
        top5
    );
    println!(
        "{backend:?} {model}: top-1 {} p={:.4}",
        got_top[0].0, got_top[0].1
    );
    true
}

#[test]
fn resnet50_matches_python_when_weights_installed() {
    for device in common::devices() {
        model_case(&device, "resnet50");
    }
}

#[test]
fn vgg16_matches_python_when_weights_installed() {
    for device in common::devices() {
        model_case(&device, "vgg16");
    }
}

#[test]
fn synthetic_forward_passes_run() {
    // Without real weights, random weights still exercise the whole orchestration
    // (shapes, buffer plumbing, kernel sequence) on every backend.
    use ndarray::Array;
    let Some(device) = common::devices().into_iter().next() else {
        return;
    };
    let dir = std::env::temp_dir().join(format!("yapocis-rs-synthetic-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("weights.npz");
    {
        use ndarray_npy::NpzWriter;
        let mut w = NpzWriter::new(std::fs::File::create(&path).unwrap());
        let mut small = |name: &str, shape: &[usize], scale: f32| {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n)
                .map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * scale)
                .collect();
            w.add_array(name, &Array::from_shape_vec(IxDyn(shape), data).unwrap())
                .unwrap();
        };
        // resnet50 layout
        small("stem_weights", &[64, 3, 7, 7], 0.05);
        small("stem_bias", &[64], 0.1);
        let blocks = [
            (0, 64, 64, 256, true),
            (1, 256, 64, 256, false),
            (2, 256, 64, 256, false),
            (3, 256, 128, 512, true),
            (4, 512, 128, 512, false),
            (5, 512, 128, 512, false),
            (6, 512, 128, 512, false),
            (7, 512, 256, 1024, true),
            (8, 1024, 256, 1024, false),
            (9, 1024, 256, 1024, false),
            (10, 1024, 256, 1024, false),
            (11, 1024, 256, 1024, false),
            (12, 1024, 256, 1024, false),
            (13, 1024, 512, 2048, true),
            (14, 2048, 512, 2048, false),
            (15, 2048, 512, 2048, false),
        ];
        for (b, i, m, o, ds) in blocks {
            small(&format!("block{b}_conv1_weights"), &[m, i, 1, 1], 0.05);
            small(&format!("block{b}_conv1_bias"), &[m], 0.1);
            small(&format!("block{b}_conv2_weights"), &[m, m, 3, 3], 0.02);
            small(&format!("block{b}_conv2_bias"), &[m], 0.1);
            small(&format!("block{b}_conv3_weights"), &[o, m, 1, 1], 0.05);
            small(&format!("block{b}_conv3_bias"), &[o], 0.1);
            if ds {
                small(&format!("block{b}_downsample_weights"), &[o, i, 1, 1], 0.05);
                small(&format!("block{b}_downsample_bias"), &[o], 0.1);
            }
        }
        small("fc_weights", &[1000, 2048], 0.01);
        small("fc_bias", &[1000], 0.1);
        w.finish().unwrap();
    }
    let net = ResNet50::load(&device, &path).unwrap();
    let input = Array::from_shape_fn((3, 224, 224), |(c, y, x)| {
        ((x + y + c) % 17) as f32 / 17.0 - 0.5
    });
    let probs = net
        .forward(&input.view(), ResNet50Layer::Probabilities)
        .unwrap();
    assert_eq!(probs.len(), 1000);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-2, "softmax sums to 1 ({sum})");
    let emb = net.forward(&input.view(), ResNet50Layer::Pool).unwrap();
    assert_eq!(emb.len(), 2048);
    assert!(emb.iter().any(|&v| v != 0.0));
    let _ = std::fs::remove_dir_all(&dir);
}
