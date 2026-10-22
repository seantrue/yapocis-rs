//! Replay the processing-library parity fixtures (`yapocis-tools bundle
//! --parity-lib ... --generators truetrace,imageops,simplex,anisodiff,fft`).

mod common;

use common::{allclose_f32, lib_index, parity_lib_dir, read_array};
use ndarray::{Array2, ArrayD, ArrayView2};
use yapocis::lib::fft::Window;
use yapocis::lib::simplex::{Params2d, Params3d};
use yapocis::lib::truetrace::{self, Curve};
use yapocis::lib::{Anisodiff, Fft, Imageops, Simplex, TrueTrace};
use yapocis::Backend;

const RTOL: f64 = 1e-4;
const ATOL: f64 = 1e-4;

fn cases(backend: Backend, generator: &str) -> Vec<serde_json::Value> {
    lib_index(backend)
        .and_then(|idx| idx["generators"][generator].as_array().cloned())
        .unwrap_or_default()
}

fn npz(backend: Backend, case: &serde_json::Value) -> ndarray_npy::NpzReader<std::fs::File> {
    common::open_npz(&parity_lib_dir(backend).join(case["file"].as_str().unwrap()))
}

fn f32s(npz: &mut ndarray_npy::NpzReader<std::fs::File>, name: &str) -> ArrayD<f32> {
    read_array::<f32>(npz, name)
}

fn check(label: &str, got: &ArrayD<f32>, want: &ArrayD<f32>) {
    assert_eq!(got.shape(), want.shape(), "{label}: shape");
    let g = got.as_standard_layout();
    let w = want.as_standard_layout();
    let (rtol, atol) = common::tolerances(RTOL, ATOL);
    if let Err(e) = allclose_f32(g.as_slice().unwrap(), w.as_slice().unwrap(), rtol, atol) {
        panic!("{label}: {e}");
    }
}

fn a2(a: ArrayD<f32>) -> Array2<f32> {
    a.into_dimensionality().unwrap()
}

/// Segments compared as sets: the GPU counter makes their order arbitrary.
fn sorted_rows(segs: &ArrayView2<'_, f32>) -> Vec<[i64; 4]> {
    let mut rows: Vec<[i64; 4]> = segs
        .rows()
        .into_iter()
        .map(|r| [0, 1, 2, 3].map(|i| (r[i] as f64 * 1000.0).round() as i64))
        .collect();
    rows.sort();
    rows
}

#[test]
fn truetrace_gpu_segments_match_python() {
    for device in common::devices() {
        let backend = device.backend();
        let tt = TrueTrace::load(&device).unwrap();
        let mut n = 0;
        for case in cases(backend, "truetrace") {
            let name = case["name"].as_str().unwrap();
            if !name.starts_with("gpu_iso") {
                continue;
            }
            let mut z = npz(backend, &case);
            let mask = read_array::<u8>(&mut z, "mask");
            let want = a2(f32s(&mut z, "segs"));
            let iso = case["iso"].as_f64().unwrap() as f32;
            let (segs, (w, h)) = tt
                .trace_segments(&mask.view().into_dimensionality().unwrap(), iso)
                .unwrap();
            assert_eq!(
                vec![w as u64, h as u64],
                case["size"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                segs.nrows(),
                want.nrows(),
                "{backend:?} {name}: segment count"
            );
            assert_eq!(
                sorted_rows(&segs.view()),
                sorted_rows(&want.view()),
                "{backend:?} {name}: segments"
            );
            println!(
                "{backend:?} {name}: {} segments (python lib agrees with per-pixel launch: {})",
                segs.nrows(),
                case["lib_matches_per_pixel"]
            );
            n += 1;
        }
        assert!(n >= 2, "{backend:?}: no truetrace GPU cases");
    }
}

#[test]
fn truetrace_cpu_pipeline_matches_python() {
    let Some(backend) = yapocis::available_backends()
        .into_iter()
        .find(|b| lib_index(*b).is_some())
    else {
        eprintln!("no library parity fixtures");
        return;
    };
    let case = cases(backend, "truetrace")
        .into_iter()
        .find(|c| c["name"] == "cpu")
        .expect("cpu case");
    let mut z = npz(backend, &case);
    let mask = read_array::<u8>(&mut z, "mask")
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();

    // signed distance field and marching squares on it (segment order is deterministic on the CPU)
    let sdf = truetrace::sdf_from_mask_cpu(&mask.view());
    check("sdf", &sdf.clone().into_dyn(), &f32s(&mut z, "sdf"));
    let ms = truetrace::marching_squares_cpu(&sdf.view(), 0.0);
    check("ms_segs", &ms.clone().into_dyn(), &f32s(&mut z, "ms_segs"));
    let field = mask.mapv(|v| v as f32);
    let ms_mask = truetrace::marching_squares_cpu(&field.view(), 0.5);
    check(
        "ms_mask_segs",
        &ms_mask.clone().into_dyn(),
        &f32s(&mut z, "ms_mask_segs"),
    );

    // chaining and simplification, in Python's order
    let polys = truetrace::chain_segments_to_polylines(&ms_mask.view(), 0.0);
    assert_eq!(
        polys.len() as u64,
        case["n_polys"].as_u64().unwrap(),
        "polyline count"
    );
    for (i, p) in polys.iter().enumerate() {
        check(
            &format!("poly_{i}"),
            &p.clone().into_dyn(),
            &f32s(&mut z, &format!("poly_{i}")),
        );
    }
    let simp: Vec<Array2<f32>> = polys
        .iter()
        .filter(|p| p.nrows() >= 3)
        .map(|p| truetrace::douglas_peucker(&p.view(), 0.75))
        .collect();
    assert_eq!(simp.len() as u64, case["n_simp"].as_u64().unwrap());
    for (i, p) in simp.iter().enumerate() {
        check(
            &format!("simp_{i}"),
            &p.clone().into_dyn(),
            &f32s(&mut z, &format!("simp_{i}")),
        );
    }

    // rings, areas, regions
    let rings = truetrace::trace_bitmap(&mask.view(), 0.75, 4.0, 0.0);
    assert_eq!(rings.len() as u64, case["n_rings"].as_u64().unwrap());
    let all = f32s(&mut z, "rings");
    let offsets = read_array::<i64>(&mut z, "ring_offsets");
    for (i, r) in rings.iter().enumerate() {
        let (a, b) = (offsets[i] as usize, offsets[i + 1] as usize);
        let want = all.slice(ndarray::s![a..b, ..]).to_owned().into_dyn();
        check(&format!("ring_{i}"), &r.clone().into_dyn(), &want);
    }
    let areas = read_array::<f64>(&mut z, "areas");
    for (i, r) in rings.iter().enumerate() {
        let a = truetrace::polygon_area(&r.view());
        assert!((a - areas[i]).abs() < 1e-3, "area {i}: {a} vs {}", areas[i]);
    }
    let regions = truetrace::build_regions_with_holes(&rings);
    let want_regions = case["regions"].as_array().unwrap();
    assert_eq!(regions.len(), want_regions.len(), "region count");
    for (i, (r, w)) in regions.iter().zip(want_regions).enumerate() {
        check(
            &format!("region_{i}_outer"),
            &r.outer.clone().into_dyn(),
            &f32s(&mut z, &format!("region_{i}_outer")),
        );
        let holes = w["holes"].as_array().unwrap();
        assert_eq!(r.holes.len(), holes.len(), "region {i} holes");
        for (j, h) in r.holes.iter().enumerate() {
            check(
                &format!("region_{i}_hole_{j}"),
                &h.clone().into_dyn(),
                &f32s(&mut z, &format!("region_{i}_hole_{j}")),
            );
        }
    }

    // SVG writers: byte-for-byte against ElementTree's output
    let dir = parity_lib_dir(backend);
    let want_svg =
        std::fs::read_to_string(dir.join(case["svg_regions"].as_str().unwrap())).unwrap();
    let svg = truetrace::svg_regions(&regions, 48, 40, 1.0, None, Curve::Cubic, 0.0);
    assert_eq!(svg, want_svg, "svg_regions");
    let want_seg_svg =
        std::fs::read_to_string(dir.join(case["svg_segments"].as_str().unwrap())).unwrap();
    let seg_svg = truetrace::svg_segments(&ms.slice(ndarray::s![..20, ..]), 48, 40);
    assert_eq!(seg_svg, want_seg_svg, "svg_segments");

    // Otsu on a synthetic two-level image
    let otsu = read_array::<f64>(&mut z, "otsu");
    let two_level = mask.mapv(|v| v * 200 + 30);
    assert_eq!(truetrace::otsu_threshold_u8(&two_level.view()), otsu[[]]);
}

#[test]
fn truetrace_masks_from_photograph() {
    let Some(backend) = yapocis::available_backends()
        .into_iter()
        .find(|b| lib_index(*b).is_some())
    else {
        return;
    };
    let data = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/lemon.jpg");
    for case in cases(backend, "truetrace") {
        let name = case["name"].as_str().unwrap();
        if !name.starts_with("lemon") {
            continue;
        }
        let mut z = npz(backend, &case);
        let gray = read_array::<u8>(&mut z, "gray")
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let want = read_array::<u8>(&mut z, "mask")
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let (invert, threshold) = (
            case["invert"].as_bool().unwrap_or(false),
            case["threshold"].as_f64(),
        );
        // the thresholding half on Pillow's grayscale is exact
        let mask = truetrace::mask_from_gray(&gray.view(), invert, threshold);
        assert_eq!(mask, want, "{name}: mask from Pillow grayscale");
        if let Some(otsu) = case["otsu"].as_f64() {
            assert_eq!(
                truetrace::otsu_threshold_u8(&gray.view()),
                otsu,
                "{name}: otsu"
            );
        }
        // decoding the JPEG ourselves differs by a level or two on some pixels
        let decoded = truetrace::load_mask_from_image(&data, invert, threshold).unwrap();
        assert_eq!(decoded.dim(), want.dim());
        let differing = decoded
            .iter()
            .zip(want.iter())
            .filter(|(a, b)| a != b)
            .count();
        let frac = differing as f64 / want.len() as f64;
        assert!(
            frac < 0.01,
            "{name}: {frac:.4} of mask pixels differ after decoding"
        );
        println!(
            "{name}: {differing} of {} pixels differ between decoders",
            want.len()
        );
    }
}

#[test]
fn imageops_matches_python() {
    for device in common::devices() {
        let backend = device.backend();
        let Some(case) = cases(backend, "imageops").into_iter().next() else {
            continue;
        };
        let ops = Imageops::load(&device).unwrap();
        let mut z = npz(backend, &case);
        let im = f32s(&mut z, "im");
        let big = f32s(&mut z, "big");
        let clipped = im.mapv(|v| v.clamp(0.0, 1.0));
        check(
            &format!("{backend:?} clamp"),
            &ops.clamp_pixels(&im.view(), 0.0, 1.0).unwrap(),
            &f32s(&mut z, "clamp"),
        );
        check(
            &format!("{backend:?} clamp_wide"),
            &ops.clamp_pixels(&im.view(), 0.1, 0.9).unwrap(),
            &f32s(&mut z, "clamp_wide"),
        );
        check(
            &format!("{backend:?} stretch"),
            &ops.stretch(&im.view(), -0.2, 1.3, 0.0, 1.0).unwrap(),
            &f32s(&mut z, "stretch"),
        );
        check(
            &format!("{backend:?} binarize"),
            &ops.binarize(&im.view(), 0.5).unwrap(),
            &f32s(&mut z, "binarize"),
        );
        let hist = ops.histogram(&clipped.view(), 256, 0.0, 1.0).unwrap();
        let want_hist = read_array::<i32>(&mut z, "histogram");
        assert_eq!(hist.into_dyn(), want_hist, "{backend:?} histogram");
        check(
            &format!("{backend:?} histeq_small"),
            &ops.histeq(&clipped.view(), 1024).unwrap(),
            &f32s(&mut z, "histeq_small"),
        );
        check(
            &format!("{backend:?} histeq_default"),
            &ops.histeq(&clipped.view(), 1 << 16).unwrap(),
            &f32s(&mut z, "histeq_default"),
        );
        check(
            &format!("{backend:?} histeq_big"),
            &ops.histeq(&big.view(), 1024).unwrap(),
            &f32s(&mut z, "histeq_big"),
        );
        let target_hist = read_array::<f64>(&mut z, "target_hist");
        check(
            &format!("{backend:?} histmatch"),
            &ops.histmatch(&clipped.view(), target_hist.as_slice().unwrap(), None)
                .unwrap(),
            &f32s(&mut z, "histmatch"),
        );
    }
}

#[test]
fn simplex_matches_python() {
    for device in common::devices() {
        let backend = device.backend();
        let Some(case) = cases(backend, "simplex").into_iter().next() else {
            continue;
        };
        let s = Simplex::load(&device).unwrap();
        let mut z = npz(backend, &case);
        let n2 = s
            .simplex2d(
                64,
                48,
                &Params2d {
                    seed: 42,
                    scale: 0.05,
                    octaves: 4,
                    ..Default::default()
                },
            )
            .unwrap();
        check(
            &format!("{backend:?} noise2d"),
            &n2.into_dyn(),
            &f32s(&mut z, "noise2d"),
        );
        let tile = s
            .simplex2d(
                64,
                48,
                &Params2d {
                    seed: 7,
                    scale: 0.1,
                    offset: (1.5, -2.0),
                    octaves: 3,
                    persistence: 0.6,
                    lacunarity: 2.5,
                    tileable: true,
                },
            )
            .unwrap();
        check(
            &format!("{backend:?} noise2d_tile"),
            &tile.into_dyn(),
            &f32s(&mut z, "noise2d_tile"),
        );
        let n3 = s
            .simplex3d(
                16,
                12,
                8,
                &Params3d {
                    seed: 99,
                    scale: 0.2,
                    offset: (0.1, 0.2, 0.3),
                    octaves: 2,
                    ..Default::default()
                },
            )
            .unwrap();
        check(
            &format!("{backend:?} noise3d"),
            &n3.into_dyn(),
            &f32s(&mut z, "noise3d"),
        );
        check(
            &format!("{backend:?} turbulence"),
            &s.turbulence(48, 32, 5, 0.08, 5).unwrap().into_dyn(),
            &f32s(&mut z, "turbulence"),
        );
        check(
            &format!("{backend:?} ridged"),
            &s.ridged(48, 32, 5, 0.08, 5).unwrap().into_dyn(),
            &f32s(&mut z, "ridged"),
        );
    }
}

#[test]
fn anisodiff_rgb_variants_match_python() {
    for device in common::devices() {
        let backend = device.backend();
        let Some(case) = cases(backend, "anisodiff").into_iter().next() else {
            continue;
        };
        let an = Anisodiff::load(&device).unwrap();
        let mut z = npz(backend, &case);
        let rgb = f32s(&mut z, "rgb");
        let gray = f32s(&mut z, "gray");
        let n = case["n"].as_u64().unwrap() as usize;
        let scale = case["scale"].as_f64().unwrap() as f32;
        let step = case["step_size"].as_f64().unwrap() as f32;
        check(
            &format!("{backend:?} gray_n"),
            &an.anisodiff_n(&gray.view(), n, scale, step).unwrap(),
            &f32s(&mut z, "gray_n"),
        );
        check(
            &format!("{backend:?} rgb_i_n"),
            &an.anisodiff_rgb_i_n(&rgb.view(), n, scale, step).unwrap(),
            &f32s(&mut z, "rgb_i_n"),
        );
        check(
            &format!("{backend:?} rgb_is_n"),
            &an.anisodiff_rgb_is_n(&rgb.view(), n, scale, step).unwrap(),
            &f32s(&mut z, "rgb_is_n"),
        );
        let mut it = an.anisodiff(&gray.view(), scale, step).unwrap();
        for i in 0..case["frames"].as_u64().unwrap() as usize {
            let frame = it.step().unwrap();
            check(
                &format!("{backend:?} frame_{i}"),
                &frame,
                &f32s(&mut z, &format!("frame_{i}")),
            );
        }
        let mut it = an.anisodiff_rgb_i(&rgb.view(), scale, step).unwrap();
        for i in 0..case["rgb_frames"].as_u64().unwrap() as usize {
            let frame = it.next().unwrap().unwrap();
            check(
                &format!("{backend:?} rgb_frame_{i}"),
                &frame,
                &f32s(&mut z, &format!("rgb_frame_{i}")),
            );
        }
        // the saturation+intensity iterator is exercised for shape only (no Python fixture)
        let mut it = an.anisodiff_rgb_is(&rgb.view(), scale, step).unwrap();
        assert_eq!(it.step().unwrap().shape(), rgb.shape());
    }
}

#[test]
fn fft_analysis_matches_python() {
    for device in common::devices() {
        let backend = device.backend();
        let Some(case) = cases(backend, "fft").into_iter().next() else {
            continue;
        };
        let fft = Fft::load(&device).unwrap();
        let mut z = npz(backend, &case);
        let sr = case["sample_rate"].as_f64().unwrap();
        let signal = f32s(&mut z, "signal")
            .into_dimensionality::<ndarray::Ix1>()
            .unwrap();
        let noise = f32s(&mut z, "noise")
            .into_dimensionality::<ndarray::Ix1>()
            .unwrap();
        let noisy = f32s(&mut z, "noisy")
            .into_dimensionality::<ndarray::Ix1>()
            .unwrap();
        let other = f32s(&mut z, "other")
            .into_dimensionality::<ndarray::Ix1>()
            .unwrap();

        let (times, freqs, spec) = fft
            .spectrogram(&noisy.view(), sr, 256, Some(64), Window::Hann, true)
            .unwrap();
        check(
            &format!("{backend:?} spec_batch_times"),
            &times.into_dyn(),
            &f32s(&mut z, "spec_batch_times"),
        );
        let want_f = read_array::<f64>(&mut z, "spec_batch_freqs");
        assert_eq!(freqs.into_dyn(), want_f, "{backend:?} spec_batch_freqs");
        // The batch FFT path (ten or more windows of at most 256 samples) is not
        // reproducible in Python either: on Metal one run in a few differs by
        // ~100 dB in some bins and on Apple's OpenCL every run differs
        // (claude-debug/fft_spectrogram_repro_check.py), so only its shape is
        // checked. The sequential path below is compared numerically.
        let want_spec = f32s(&mut z, "spec_batch");
        assert_eq!(
            spec.shape(),
            want_spec.shape(),
            "{backend:?} spec_batch shape"
        );

        let (times, freqs, spec) = fft
            .spectrogram(
                &noisy.slice(ndarray::s![..1500]),
                sr,
                512,
                Some(256),
                Window::Hann,
                false,
            )
            .unwrap();
        check(
            &format!("{backend:?} spec_seq_times"),
            &times.into_dyn(),
            &f32s(&mut z, "spec_seq_times"),
        );
        assert_eq!(
            freqs.into_dyn(),
            read_array::<f64>(&mut z, "spec_seq_freqs")
        );
        check(
            &format!("{backend:?} spec_seq"),
            &spec.into_dyn(),
            &f32s(&mut z, "spec_seq"),
        );

        let bp = fft
            .bandpass_filter(&noisy.slice(ndarray::s![..256]), 300.0, 600.0, sr)
            .unwrap();
        check(
            &format!("{backend:?} bandpass"),
            &bp.into_dyn(),
            &f32s(&mut z, "bandpass"),
        );

        let (cf, coh) = fft
            .coherence(
                &noisy.slice(ndarray::s![..256]),
                &other.slice(ndarray::s![..256]),
                Window::Hann,
            )
            .unwrap();
        assert_eq!(cf.into_dyn(), read_array::<f64>(&mut z, "coherence_freqs"));
        check(
            &format!("{backend:?} coherence"),
            &coh.into_dyn(),
            &f32s(&mut z, "coherence"),
        );

        let est = fft
            .estimate_frequency(&noisy.slice(ndarray::s![..256]), sr, Window::Hann)
            .unwrap();
        let want_est = read_array::<f64>(&mut z, "estimate_frequency");
        assert_eq!(est, want_est[[]], "{backend:?} estimate_frequency");

        let scalar = |z: &mut ndarray_npy::NpzReader<std::fs::File>, name: &str| -> f64 {
            let a = read_array::<f64>(z, name);
            a[[]]
        };
        let rel = |a: f64, b: f64| (a - b).abs() <= 1e-5 * b.abs().max(1.0);
        assert!(
            rel(
                Fft::energy(noisy.as_slice().unwrap()),
                scalar(&mut z, "energy")
            ),
            "{backend:?} energy"
        );
        assert!(
            rel(Fft::rms(noisy.as_slice().unwrap()), scalar(&mut z, "rms")),
            "{backend:?} rms"
        );
        assert!(
            rel(
                Fft::snr(signal.as_slice().unwrap(), noise.as_slice().unwrap()),
                scalar(&mut z, "snr")
            ),
            "{backend:?} snr"
        );
    }
}

/// The potrace replacement: `fill_regions` and `trace_mask` rasters equal
/// Python's bit for bit (the `fill` case of the truetrace generator).
#[test]
fn truetrace_fill_matches_python() {
    use ndarray::Array2;
    for device in common::devices() {
        let backend = device.backend();
        for case in cases(backend, "truetrace") {
            if case["name"] != "fill" {
                continue;
            }
            let mut z = npz(backend, &case);
            let mask = read_array::<u8>(&mut z, "mask");
            let mask: Array2<u8> = mask.into_dimensionality().unwrap();
            let rings = truetrace::trace_bitmap(&mask.view(), 0.0, 0.0, 0.0);
            let regions = truetrace::build_regions_with_holes(&rings);
            assert_eq!(
                regions.len() as u64,
                case["regions"].as_u64().unwrap(),
                "{backend:?}: regions"
            );
            let exact = |label: &str, got: &Array2<f32>, want: &ArrayD<f32>| {
                let want: ndarray::ArrayView2<f32> = want.view().into_dimensionality().unwrap();
                assert_eq!(got.dim(), want.dim(), "{backend:?} {label}: shape");
                let diff = got
                    .iter()
                    .zip(want.iter())
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(diff, 0, "{backend:?} {label}: {diff} pixels differ");
            };
            let none =
                truetrace::fill_regions(&regions, mask.dim(), "none", 0.0, 8, 0.0, 1.0).unwrap();
            exact("filled_none", &none, &f32s(&mut z, "filled_none"));
            let cubic =
                truetrace::fill_regions(&regions, mask.dim(), "cubic", 0.0, 8, 0.0, 1.0).unwrap();
            exact("filled_cubic", &cubic, &f32s(&mut z, "filled_cubic"));
            for engine in ["gpu", "cpu"] {
                for iso in [0.0f32, -1.0] {
                    let params = truetrace::TraceParams {
                        iso,
                        engine: engine.to_string(),
                        ..Default::default()
                    };
                    let got = truetrace::trace_mask(&device, &mask.view(), &params).unwrap();
                    let key = format!("trace_{engine}_iso{iso:.1}");
                    exact(&key, &got, &f32s(&mut z, &key));
                }
            }
            // 1 px strokes traced inside the ink (the chaining fix of 2026-10-21)
            let thin = read_array::<u8>(&mut z, "thin_mask");
            let thin: Array2<u8> = thin.into_dimensionality().unwrap();
            for engine in ["gpu", "cpu"] {
                for iso in [0.75f32, 1.0] {
                    let params = truetrace::TraceParams {
                        iso,
                        curve: "none".to_string(),
                        engine: engine.to_string(),
                        ..Default::default()
                    };
                    let got = truetrace::trace_mask(&device, &thin.view(), &params).unwrap();
                    let key = format!(
                        "thin_{engine}_iso{}",
                        if iso == 1.0 {
                            "1.0".to_string()
                        } else {
                            iso.to_string()
                        }
                    );
                    exact(&key, &got, &f32s(&mut z, &key));
                }
            }
            println!("{backend:?}: fill case matches");
        }
    }
}
