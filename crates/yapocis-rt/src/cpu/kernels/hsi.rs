//! `hsi.cl`: RGB to hue/saturation/intensity and back (the GLSL HSV
//! formulation), on planar or interleaved images, plus `rgb2lab3`.

use crate::cpu::{par, Args};

pub const KERNELS: super::Table = &[
    ("rgb2hsi", rgb2hsi),
    ("hsi2rgb", hsi2rgb),
    ("rgb2hsi3", rgb2hsi3),
    ("hsi2rgb3", hsi2rgb3),
    ("rgb2lab3", rgb2lab3),
];

/// `eps` is `1.0e-10`, a `double` literal.
const EPS: f64 = 1.0e-10;

/// `_rgb2hsi(float3 c)`. Callers pass `c = (r, b, g)`: green and blue are
/// swapped on the way in, as in the OpenCL.
fn rgb_to_hsi([cx, cy, cz]: [f32; 3]) -> [f32; 3] {
    const K: [f32; 4] = [0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0];
    let p = if cy < cz {
        [cz, cy, K[3], K[2]]
    } else {
        [cy, cz, K[0], K[1]]
    };
    let q = if cx < p[0] {
        [p[0], p[1], p[3], cx]
    } else {
        [cx, p[1], p[2], p[0]]
    };
    let d = q[0] - q[3].min(q[1]);
    // `eps` makes these two divisions double precision.
    let hh = (q[2] as f64 + (q[3] - q[1]) as f64 / (6.0 * d as f64 + EPS)) as f32;
    let hh = if hh < 0.0 { -hh } else { hh };
    let s = (d as f64 / (q[0] as f64 + EPS)) as f32;
    // clamp(hsi, 0.0f, 1.0f - eps): the upper bound rounds to 1.0f.
    [hh, s, q[0]].map(|v| v.clamp(0.0, 1.0))
}

/// OpenCL `fract`: `x - floor(x)`, kept below 1.
fn fract(x: f32) -> f32 {
    (x - x.floor()).min(f32::from_bits(0x3f7f_ffff))
}

/// `_hsi2rgb(float3 hsi)`.
fn hsi_to_rgb([h, s, i]: [f32; 3]) -> [f32; 3] {
    const K: [f32; 4] = [1.0, 1.0 / 3.0, 2.0 / 3.0, 3.0];
    let channel = |k: f32| {
        let p = (fract(h + k) * 6.0 - K[3]).abs();
        let c = (p - K[0]).clamp(0.0, 1.0);
        // mix(1, c, s) = 1 + (c - 1) * s
        i * (K[0] + (c - K[0]) * s)
    };
    [channel(K[0]), channel(K[1]), channel(K[2])]
}

/// `rgb2hsi(float* r, float* g, float* b, float* h, float* s, float* i)`.
fn rgb2hsi(a: &Args) {
    planar(a, |[r, g, b]| rgb_to_hsi([r, b, g]));
}

/// `hsi2rgb(float* h, float* s, float* i, float* r, float* g, float* b)`.
fn hsi2rgb(a: &Args) {
    planar(a, hsi_to_rgb);
}

/// Three planar inputs (arguments 0..3) to three planar outputs (3..6).
fn planar(a: &Args, f: impl Fn([f32; 3]) -> [f32; 3] + Sync) {
    let (x, y, z) = (a.input::<f32>(0), a.input::<f32>(1), a.input::<f32>(2));
    let n = a.global_size();
    let mut out = vec![[0f32; 3]; n];
    par::map(&mut out, |i| f([x[i], y[i], z[i]]));
    let (mut o0, mut o1, mut o2) = (a.output::<f32>(3), a.output::<f32>(4), a.output::<f32>(5));
    for (i, [p, q, r]) in out.into_iter().enumerate() {
        (o0[i], o1[i], o2[i]) = (p, q, r);
    }
}

/// Interleaved pixels of `depth` channels: `(int depth, float* input, float*
/// output)`. Work item `g` converts the first three channels of pixel `g`
/// and, when `depth > 3`, copies channel 3 (alpha) through. Other channels
/// of the output are left as they were.
fn interleaved(a: &Args, f: impl Fn(&[f32]) -> [f32; 3] + Sync) {
    let depth = a.int(0) as usize;
    let input = a.input::<f32>(1);
    let mut output = a.output::<f32>(2);
    let n = a.global_size();
    par::map_chunks(&mut output[..n * depth], depth, |g, o| {
        let px = &input[g * depth..g * depth + depth];
        o[..3].copy_from_slice(&f(px));
        if depth > 3 {
            o[3] = px[3];
        }
    });
}

/// `rgb2hsi3(int depth, float* input, float* output)`; green and blue are
/// swapped on the way in.
fn rgb2hsi3(a: &Args) {
    interleaved(a, |p| rgb_to_hsi([p[0], p[2], p[1]]));
}

/// `hsi2rgb3(int depth, float* input, float* output)`.
fn hsi2rgb3(a: &Args) {
    interleaved(a, |p| hsi_to_rgb([p[0], p[1], p[2]]));
}

/// `rgb2lab3(int depth, float* input, float* output)`: sRGB to CIE L*a*b*
/// (D65); green and blue are swapped on the way in, and channel 3 is not
/// copied. Not in the interface, ported for completeness.
fn rgb2lab3(a: &Args) {
    let depth = a.int(0) as usize;
    let input = a.input::<f32>(1);
    let mut output = a.output::<f32>(2);
    let n = a.global_size();
    par::map_chunks(&mut output[..n * depth], depth, |g, o| {
        let p = &input[g * depth..g * depth + 3];
        o[..3].copy_from_slice(&xyz_to_lab(rgb_to_xyz([p[0], p[2], p[1]])));
    });
}

/// `adjustrgb`: sRGB gamma expansion, times 100 (a `double` literal).
fn adjust_rgb(v: f32) -> f32 {
    let v = if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    };
    (v as f64 * 100.0) as f32
}

/// `_rgb2xyz`.
fn rgb_to_xyz(rgb: [f32; 3]) -> [f32; 3] {
    const XK: [f32; 3] = [0.4124, 0.3576, 0.1805];
    const YK: [f32; 3] = [0.2126, 0.7152, 0.0722];
    const ZK: [f32; 3] = [0.0193, 0.1192, 0.9505];
    const WHITE: [f32; 3] = [95.047, 100.0, 108.883];
    let v = rgb.map(adjust_rgb);
    let dot3 = |k: [f32; 3]| v[0] * k[0] + v[1] * k[1] + v[2] * k[2];
    [
        dot3(XK) / WHITE[0],
        dot3(YK) / WHITE[1],
        dot3(ZK) / WHITE[2],
    ]
}

/// `adjustxyz`.
fn adjust_xyz(v: f32) -> f32 {
    if v <= 0.008856 {
        7.787 * v + 16.0 / 116.0
    } else {
        v.powf(0.333_333_34)
    }
}

/// `_xyz2lab`.
fn xyz_to_lab(xyz: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = xyz.map(adjust_xyz);
    [116.0 * y - 16.0, 500.0 * (x - y), 200.0 * (y - z)]
}
