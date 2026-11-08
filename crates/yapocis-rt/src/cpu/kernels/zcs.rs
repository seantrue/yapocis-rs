//! `zcs.cl`: zero crossings of images with 1 to 4 channels per pixel, and
//! a pixel-wise subtraction.

use crate::cpu::{par, Args};

pub const KERNELS: super::Table = &[
    ("zcs1", zcs1),
    ("zcs2", zcs2),
    ("zcs3", zcs3),
    ("zcs4", zcs4),
    ("sub", sub),
];

/// `zcsN(int width, int height, float threshold, float* input, float* zcs)`
fn zcs1(a: &Args) {
    zcs(a, 1)
}

fn zcs2(a: &Args) {
    zcs(a, 2)
}

fn zcs3(a: &Args) {
    zcs(a, 3)
}

fn zcs4(a: &Args) {
    zcs(a, 4)
}

/// OpenCL `sign`: 1, -1, or the zero itself (NaN gives 0).
fn sign(v: f32) -> f32 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else if v.is_nan() {
        0.0
    } else {
        v
    }
}

/// Work item `g` writes `zcs[g*D .. g*D + D]`: 1 where the squared central
/// difference exceeds `threshold` and the neighbours above/below or
/// left/right differ in sign, else 0.
///
/// Quirks kept from the kernel: `x` and `y` come from the flat float index
/// `g * D` (not the pixel index `g`), clamped to `1..=width-2` and
/// `1..=height-2`; and every channel reads the neighbours at channel 0
/// (`ADDRESS(x, y) * D`), so all `D` outputs of a work item are equal.
fn zcs(args: &Args, depth: usize) {
    let width = args.int(0) as i64;
    let height = args.int(1) as i64;
    let threshold = args.float(2);
    let input = args.input::<f32>(3);
    let mut out = args.output::<f32>(4);
    let at = |x: i64, y: i64| input[((x * height + y) as usize) * depth];
    par::map_chunks(&mut out[..args.global_size() * depth], depth, |g, out| {
        let i = (g * depth) as i64;
        let x = (i / height).max(1).min(width - 2);
        let y = (i % height).max(1).min(height - 2);
        let (above, below) = (at(x, y + 1), at(x, y - 1));
        let (left, right) = (at(x - 1, y), at(x + 1, y));
        let delta = (above - below) * (above - below) + (right - left) * (right - left);
        let crossing = sign(below) != sign(above) || sign(left) != sign(right);
        let z = if delta > threshold && crossing {
            1.0
        } else {
            0.0
        };
        out.fill(z);
    });
}

/// `sub(float* input1, float* input2, float* output)`
fn sub(a: &Args) {
    let input1 = a.input::<f32>(0);
    let input2 = a.input::<f32>(1);
    let mut output = a.output::<f32>(2);
    par::map(&mut output[..a.global_size()], |i| input1[i] - input2[i]);
}
