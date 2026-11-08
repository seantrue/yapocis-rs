//! `gradient.cl`: gradient magnitude and direction of images with 1 to 4
//! channels per pixel.
//!
//! The image is addressed `i = x * height + y`; a pixel's channels are
//! contiguous. An OpenCL `float3` occupies 16 bytes in a buffer, so
//! `gradient3` steps 4 floats per pixel and leaves the fourth untouched.

use crate::cpu::{par, Args};

pub const KERNELS: super::Table = &[
    ("gradient", gradient1),
    ("gradient2", gradient2),
    ("gradient3", gradient3),
    ("gradient4", gradient4),
];

/// `gradient(int width, int height, float* input, int reach, float* grad, float* theta)`
fn gradient1(a: &Args) {
    gradient(a, 1, 1)
}

/// `gradient2(..., float2* input, int reach, float2* grad, float2* theta)`
fn gradient2(a: &Args) {
    gradient(a, 2, 2)
}

/// `gradient3(..., float3* input, int reach, float3* grad, float3* theta)`
fn gradient3(a: &Args) {
    gradient(a, 3, 4)
}

/// `gradient4(..., float4* input, int reach, float4* grad, float4* theta)`
fn gradient4(a: &Args) {
    gradient(a, 4, 4)
}

/// Central differences `reach` pixels each way: `grad = sqrt(dx² + dy²)` and
/// `theta = atan2(dy, dx) / π`, both 0 within `reach` of an edge and where
/// they are NaN. `stride` is the number of floats a pixel occupies.
fn gradient(args: &Args, channels: usize, stride: usize) {
    let width = args.int(0) as i64;
    let height = args.int(1) as i64;
    let input = args.input::<f32>(2);
    let reach = args.int(3) as i64;
    let n = args.global_size() * stride;

    // (dx, dy) of channel `c` at pixel `i`, or None outside the margin.
    let differences = |i: usize, c: usize| {
        let (x, y) = (i as i64 / height, i as i64 % height);
        if !(0 <= x - reach && x + reach < width && 0 <= y - reach && y + reach < height) {
            return None;
        }
        let at = |x: i64, y: i64| input[(x * height + y) as usize * stride + c];
        Some((
            at(x + reach, y) - at(x - reach, y),
            at(x, y + reach) - at(x, y - reach),
        ))
    };
    let zero_nan = |v: f32| if v.is_nan() { 0.0 } else { v };

    // Each output is written by its own pass; both read the same differences.
    let mut grad = args.output::<f32>(4);
    par::map_chunks(&mut grad[..n], stride, |i, out| {
        for (c, o) in out.iter_mut().enumerate().take(channels) {
            let gr = differences(i, c).map_or(0.0, |(dx, dy)| (dx * dx + dy * dy).sqrt());
            *o = zero_nan(gr);
        }
    });
    drop(grad);
    let mut theta = args.output::<f32>(5);
    par::map_chunks(&mut theta[..n], stride, |i, out| {
        for (c, o) in out.iter_mut().enumerate().take(channels) {
            let th = differences(i, c).map_or(0.0, |(dx, dy)| dy.atan2(dx) / std::f32::consts::PI);
            *o = zero_nan(th);
        }
    });
}
