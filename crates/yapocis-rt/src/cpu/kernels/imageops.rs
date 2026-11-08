//! `imageops.cl`: clamping, contrast stretch, thresholding and a histogram.

use crate::cpu::{par, Args};

pub const KERNELS: super::Table = &[
    ("clamp_pixels", clamp_pixels),
    ("stretch", stretch),
    ("binarize", binarize),
    ("histogram", histogram),
];

/// `low` / `high` bounds as the OpenCL's two ternaries apply them (a NaN
/// passes through).
fn clamp(value: f32, low: f32, high: f32) -> f32 {
    let value = if value < low { low } else { value };
    if value > high {
        high
    } else {
        value
    }
}

/// `clamp_pixels(float* input, float low, float high, float* output)`.
fn clamp_pixels(a: &Args) {
    let input = a.input::<f32>(0);
    let (low, high) = (a.float(1), a.float(2));
    let mut output = a.output::<f32>(3);
    par::map(&mut output[..a.global_size()], |i| {
        clamp(input[i], low, high)
    });
}

/// `stretch(float* input, float input_min, float input_max, float low, float
/// high, float* output)`: clamp to the input range, then map it linearly
/// onto `low..high`.
fn stretch(a: &Args) {
    let input = a.input::<f32>(0);
    let (input_min, input_max) = (a.float(1), a.float(2));
    let (low, high) = (a.float(3), a.float(4));
    let mut output = a.output::<f32>(5);
    par::map(&mut output[..a.global_size()], |i| {
        let value = clamp(input[i], input_min, input_max);
        let value = (value - input_min) / (input_max - input_min);
        low + value * (high - low)
    });
}

/// `binarize(float* input, float threshold, float* output)`: 1 where
/// `input >= threshold`, else 0.
fn binarize(a: &Args) {
    let input = a.input::<f32>(0);
    let threshold = a.float(1);
    let mut output = a.output::<f32>(2);
    par::map(&mut output[..a.global_size()], |i| {
        if input[i] >= threshold {
            1.0
        } else {
            0.0
        }
    });
}

/// `histogram(float* input, int size_of_input, float low, float high, int*
/// hst, int size_of_hst)`: adds each clamped value to one of `size_of_hst`
/// bins spanning `low..=high` (a value equal to `high` goes in the last bin).
/// The bins are incremented, not reset.
///
/// Work items share bins (`atomic_inc` on the GPU). Here each chunk of the
/// input is counted into its own partial histogram, the chunks in parallel,
/// and the partials are added into `hst` in chunk order. The counts are
/// integers, so the result is exactly the GPU's, whatever the order.
fn histogram(a: &Args) {
    let input = a.input::<f32>(0);
    let (low, high) = (a.float(2), a.float(3));
    let mut hst = a.output::<i32>(4);
    let n_bins = a.int(5) as usize;
    let values = &input[..a.global_size()];
    let bucket = |x: f32| {
        let value = clamp(x, low, high);
        // (size_t) of the float quotient; `as` truncates toward zero as C does
        // for in-range values (and gives 0 for NaN).
        let bucket = ((n_bins as f32 * (value - low)) / (high - low)) as usize;
        bucket.min(n_bins - 1)
    };
    const CHUNK: usize = 1 << 14;
    let mut partials = vec![Vec::new(); values.len().div_ceil(CHUNK)];
    par::map_each(&mut partials, |k| {
        let mut counts = vec![0i32; n_bins];
        for &x in &values[k * CHUNK..((k + 1) * CHUNK).min(values.len())] {
            counts[bucket(x)] += 1;
        }
        counts
    });
    for counts in &partials {
        for (bin, count) in hst.iter_mut().zip(counts) {
            *bin += count;
        }
    }
}
