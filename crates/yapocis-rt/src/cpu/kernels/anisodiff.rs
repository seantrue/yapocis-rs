//! `anisodiff.cl`: one step of Perona-Malik anisotropic diffusion for images
//! of 1 to 4 channels per pixel.
//!
//! The image is addressed `i = x * height + y`; a pixel's channels are
//! contiguous (`float`, `float2`, `float3`, `float4`). An OpenCL `float3`
//! occupies 16 bytes in a buffer, so `filterImage3` steps 4 floats per pixel
//! and leaves the fourth untouched. The kernels end with a `barrier`, which
//! orders nothing here (no local memory is used).

use crate::cpu::{par, Args};

pub const KERNELS: super::Table = &[
    ("filterImage", filter_image1),
    ("filterImage2", filter_image2),
    ("filterImage3", filter_image3),
    ("filterImage4", filter_image4),
];

/// `filterImage(int width, int height, float* input, float scale, float step_size, float* output)`
fn filter_image1(a: &Args) {
    filter_image(a, 1, 1)
}

/// `filterImage2(..., float2* input, ..., float2* output)`
fn filter_image2(a: &Args) {
    filter_image(a, 2, 2)
}

/// `filterImage3(..., float3* input, ..., float3* output)`
fn filter_image3(a: &Args) {
    filter_image(a, 3, 4)
}

/// `filterImage4(..., float4* input, ..., float4* output)`
fn filter_image4(a: &Args) {
    filter_image(a, 4, 4)
}

/// Work item `i` (one pixel) moves each channel towards its four neighbours,
/// each weighted by `1 / (1 + (d / scale)^2)`. Edge pixels are copied.
/// `stride` is the number of floats a pixel occupies in the buffer.
fn filter_image(args: &Args, channels: usize, stride: usize) {
    let width = args.int(0) as i64;
    let height = args.int(1) as i64;
    let input = args.input::<f32>(2);
    let scale = args.float(3);
    let step_size = args.float(4);
    let mut output = args.output::<f32>(5);
    let invscale = 1.0f32 / scale;
    let h = height as usize;
    par::map_chunks(
        &mut output[..args.global_size() * stride],
        stride,
        |i, out| {
            let (x, y) = (i as i64 / height, i as i64 % height);
            let inside = 0 < x && x < width - 1 && 0 < y && y < height - 1;
            for c in 0..channels {
                let at = |p: usize| input[p * stride + c];
                let mut central = at(i);
                if inside {
                    // north, south, east, west, in the kernel's order
                    let neighbours = [at(i + 1), at(i - 1), at(i - h), at(i + h)];
                    let mut accumulator = 0f32;
                    for (k, &n) in neighbours.iter().enumerate() {
                        let di = n - central;
                        let cf = di * invscale;
                        let cf = 1.0 / (1.0 + cf * cf);
                        if k == 0 {
                            accumulator = di * cf;
                        } else {
                            accumulator += di * cf;
                        }
                    }
                    accumulator *= step_size;
                    central += accumulator;
                }
                out[c] = central;
            }
        },
    );
}
