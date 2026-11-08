//! `audio.cl`: spectrogram features and smoothing for voice-activity detection.
//! Spectrograms are row-major `(freq, time)`: element `(f, t)` is at `f * width + t`.

use crate::cpu::{par, Args};

pub const KERNELS: super::Table = &[
    ("spectral_variance", spectral_variance),
    ("log_normalize", log_normalize),
    ("normalize_with_minmax", normalize_with_minmax),
    ("fused_vad_features", fused_vad_features),
    ("smooth_1d", smooth_1d),
];

/// Work items `t` of a per-time-slice kernel that pass `t < width`.
fn time_slices(a: &Args, width: i32) -> usize {
    a.global_size().min(width.max(0) as usize)
}

/// `spectral_variance(int width, int height, float* spectrogram, float*
/// variance)`: the variance over frequency of each time slice. Not in the
/// interface, ported for completeness.
fn spectral_variance(a: &Args) {
    let (width, height) = (a.int(0), a.int(1));
    let spec = a.input::<f32>(2);
    let mut variance = a.output::<f32>(3);
    let w = width as usize;
    let n = time_slices(a, width);
    par::map(&mut variance[..n], |t| {
        let column = (0..height as usize).map(|f| spec[f * w + t]);
        let mean = column.clone().fold(0f32, |s, v| s + v) / height as f32;
        column.fold(0f32, |s, v| s + (v - mean) * (v - mean)) / height as f32
    });
}

/// `log_normalize(float* input, float* output)`: `log(x + 1e-10f)`.
fn log_normalize(a: &Args) {
    let input = a.input::<f32>(0);
    let mut output = a.output::<f32>(1);
    par::map(&mut output[..a.global_size()], |i| (input[i] + 1e-10).ln());
}

/// `normalize_with_minmax(float* input, float min_val, float max_val, float*
/// output)`: `(x - min) / (max - min)`, or 0 when the range is at most `1e-10`.
fn normalize_with_minmax(a: &Args) {
    let input = a.input::<f32>(0);
    let (min_val, max_val) = (a.float(1), a.float(2));
    let mut output = a.output::<f32>(3);
    let range = max_val - min_val;
    par::map(&mut output[..a.global_size()], |i| {
        if range > 1e-10 {
            (input[i] - min_val) / range
        } else {
            0.0
        }
    });
}

/// `fused_vad_features(float* spectrogram, int lower_freq_idx, int
/// upper_freq_idx, int width, int height, float* features)`: the log of the
/// variance over the band `lower..upper` of each time slice. Rows outside
/// `0..height` are skipped but still counted in the band height.
fn fused_vad_features(a: &Args) {
    let spec = a.input::<f32>(0);
    let (lower, upper, width, height) = (a.int(1), a.int(2), a.int(3), a.int(4));
    let mut features = a.output::<f32>(5);
    let n = time_slices(a, width);
    let band_height = upper - lower;
    let w = width as usize;
    par::map(&mut features[..n], |t| {
        if band_height <= 0 {
            return 0.0;
        }
        let band = (lower..upper)
            .filter(|&f| f >= 0 && f < height)
            .map(|f| spec[f as usize * w + t]);
        let mean = band.clone().fold(0f32, |s, v| s + v) / band_height as f32;
        let variance = band.fold(0f32, |s, v| s + (v - mean) * (v - mean)) / band_height as f32;
        (variance + 1e-10).ln()
    });
}

/// `smooth_1d(float* input, int window_size, int n_param, float* output)`: a
/// box filter of `window_size / 2` on each side, clamping indices to the
/// array, whose length is the global size (`n_param` is for Metal).
fn smooth_1d(a: &Args) {
    let input = a.input::<f32>(0);
    let half_window = a.int(1) / 2;
    let mut output = a.output::<f32>(3);
    let n = a.global_size() as i64;
    par::map(&mut output[..n as usize], |i| {
        let mut sum = 0f32;
        let mut count = 0;
        for offset in -half_window..=half_window {
            let idx = (i as i64 + offset as i64).clamp(0, n - 1);
            sum += input[idx as usize];
            count += 1;
        }
        sum / count as f32
    });
}
