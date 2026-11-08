//! `cnn.cl`: convolution, activation, pooling, normalisation, dense and
//! regularisation layers on channel-major (`C x H x W`) float tensors.
//!
//! Most kernels are one work item per output element at
//! `c * height * width + h * width + w`, guarded by `h < height`, `w < width`,
//! `c < channels`. `per_output` runs them by output element instead: an
//! element is computed only if the work item that writes it was launched
//! (its ids are below the global size) and passes the guard, so elements no
//! work item writes keep their contents, as on the GPU.
//!
//! `uint` arguments are read as `u32`; index arithmetic is `usize`.

use crate::cpu::{par, Args};

pub const KERNELS: super::Table = &[
    ("conv2d_3x3", conv2d_3x3),
    ("conv2d_3x3_optimized", conv2d_3x3),
    ("conv2d_1x1", conv2d_1x1),
    ("depthwise_conv2d_3x3", depthwise_conv2d_3x3),
    ("conv2d_7x7_stride2", conv2d_7x7_stride2),
    ("conv2d_3x3_stride2", conv2d_3x3_stride2),
    ("conv2d_1x1_stride2", conv2d_1x1_stride2),
    ("relu_activation", relu_activation),
    ("gelu", gelu),
    ("softmax", softmax),
    ("sigmoid", sigmoid),
    ("max_pool_2x2", max_pool_2x2),
    ("max_pool_3x3_stride2", max_pool_3x3_stride2),
    ("global_avg_pool", global_avg_pool),
    ("global_max_pool", global_max_pool),
    ("batch_norm", batch_norm),
    ("layer_norm", layer_norm),
    ("layer_norm_1d", layer_norm_1d),
    ("fully_connected", fully_connected),
    ("matrix_multiply_simd", matrix_multiply_simd),
    ("add_tensors", add_tensors),
    ("dropout", dropout),
    ("spatial_dropout", spatial_dropout),
    ("upsample_2x", upsample_2x),
];

fn uint(a: &Args, i: usize) -> usize {
    a.scalar::<u32>(i) as usize
}

/// Which global axis carries `h`, `w` and `c`. Most kernels launch
/// `(h, w, c)`; the strided convolutions and `upsample_2x` launch `(h, c, w)`.
#[derive(Clone, Copy)]
pub(super) struct Axes {
    h: usize,
    w: usize,
    c: usize,
}

pub(super) const HWC: Axes = Axes { h: 0, w: 1, c: 2 };
const HCW: Axes = Axes { h: 0, w: 2, c: 1 };

/// Fill `out[c * height * width + h * width + w] = f(h, w, c)` for every
/// `h < height`, `w < width`, `c < channels` whose work item was launched.
pub(super) fn per_output(
    a: &Args,
    out: &mut [f32],
    (height, width, channels): (usize, usize, usize),
    axes: Axes,
    f: impl Fn(usize, usize, usize) -> f32 + Sync,
) {
    let plane = height * width;
    let launched = |h: usize, w: usize, c: usize| {
        h < a.global_dim(axes.h) && w < a.global_dim(axes.w) && c < a.global_dim(axes.c)
    };
    par::map_chunks(&mut out[..channels * plane], 1, |idx, o| {
        let (c, h, w) = (idx / plane, idx % plane / width, idx % width);
        if launched(h, w, c) {
            o[0] = f(h, w, c);
        }
    });
}

/// `out[gid] = f(gid)` for `gid < size` among the launched work items.
pub(super) fn per_item(a: &Args, out: &mut [f32], size: usize, f: impl Fn(usize) -> f32 + Sync) {
    let n = size.min(a.global_size());
    par::map(&mut out[..n], f);
}

/// A square convolution's inputs: `input` is `[in_c][in_h][in_w]` and
/// `weights` `[oc][in_c][k][k]`.
struct Conv<'a> {
    input: &'a [f32],
    weights: &'a [f32],
    bias: &'a [f32],
    in_h: usize,
    in_w: usize,
    in_c: usize,
    k: usize,
}

impl Conv<'_> {
    /// Output channel `oc` with the window's top-left at `(h0, w0)`:
    /// `bias[oc]` plus `input * weight` over `(ic, kh, kw)` in that order,
    /// skipping taps outside the input (zero padding).
    fn tap_sum(&self, oc: usize, h0: i64, w0: i64) -> f32 {
        let (in_h, in_w, in_c, k) = (self.in_h, self.in_w, self.in_c, self.k);
        let mut sum = self.bias[oc];
        for ic in 0..in_c {
            for kh in 0..k {
                for kw in 0..k {
                    let (y, x) = (h0 + kh as i64, w0 + kw as i64);
                    if y >= 0 && y < in_h as i64 && x >= 0 && x < in_w as i64 {
                        let input_idx = ic * in_h * in_w + y as usize * in_w + x as usize;
                        let weight_idx = ((oc * in_c + ic) * k + kh) * k + kw;
                        sum += self.input[input_idx] * self.weights[weight_idx];
                    }
                }
            }
        }
        sum
    }
}

/// The convolution kernels' common arguments: `(input, weights, bias, output,
/// uint input_height, uint input_width, uint input_channels, uint output_channels)`.
/// Runs `f(conv, output, output_channels)`.
fn with_conv(a: &Args, k: usize, f: impl FnOnce(&Conv, &mut [f32], usize)) {
    let input = a.input::<f32>(0);
    let weights = a.input::<f32>(1);
    let bias = a.input::<f32>(2);
    let mut output = a.output::<f32>(3);
    let conv = Conv {
        input: &input,
        weights: &weights,
        bias: &bias,
        in_h: uint(a, 4),
        in_w: uint(a, 5),
        in_c: uint(a, 6),
        k,
    };
    f(&conv, &mut output, uint(a, 7));
}

/// `conv2d_3x3(input, weights, bias, output, uint input_height, uint input_width,
/// uint input_channels, uint output_channels)`: padding 1, stride 1.
/// `conv2d_3x3_optimized` is the same loop written with `kh, kw` in `-1..=1`.
pub(super) fn conv2d_3x3(a: &Args) {
    with_conv(a, 3, |conv, output, oc| {
        per_output(a, output, (conv.in_h, conv.in_w, oc), HWC, |h, w, c| {
            conv.tap_sum(c, h as i64 - 1, w as i64 - 1)
        });
    });
}

/// `conv2d_1x1(...)`, same arguments as `conv2d_3x3`.
fn conv2d_1x1(a: &Args) {
    with_conv(a, 1, |conv, output, oc| {
        per_output(a, output, (conv.in_h, conv.in_w, oc), HWC, |h, w, c| {
            conv.tap_sum(c, h as i64, w as i64)
        });
    });
}

/// `depthwise_conv2d_3x3(input, weights, bias, output, uint input_height,
/// uint input_width, uint channels)`: each channel convolved with its own 3x3.
fn depthwise_conv2d_3x3(a: &Args) {
    let input = a.input::<f32>(0);
    let weights = a.input::<f32>(1);
    let bias = a.input::<f32>(2);
    let mut output = a.output::<f32>(3);
    let (ih, iw, ch) = (uint(a, 4), uint(a, 5), uint(a, 6));
    let plane = ih * iw;
    per_output(a, &mut output, (ih, iw, ch), HWC, |h, w, c| {
        // A one-channel convolution: the channel's plane, its 9 weights and its bias.
        let conv = Conv {
            input: &input[c * plane..(c + 1) * plane],
            weights: &weights[c * 9..(c + 1) * 9],
            bias: &bias[c..c + 1],
            in_h: ih,
            in_w: iw,
            in_c: 1,
            k: 3,
        };
        conv.tap_sum(0, h as i64 - 1, w as i64 - 1)
    });
}

/// `conv2d_7x7_stride2(...)`: padding 3, output `input / 2`; launched `(h, c, w)`.
fn conv2d_7x7_stride2(a: &Args) {
    strided_conv(a, 7, 3, |n| n / 2);
}

/// `conv2d_3x3_stride2(...)`: padding 1, output `(input - 1) / 2 + 1`; launched `(h, c, w)`.
fn conv2d_3x3_stride2(a: &Args) {
    strided_conv(a, 3, 1, |n| n.wrapping_sub(1) / 2 + 1);
}

/// `conv2d_1x1_stride2(...)`: every second pixel, output `input / 2`; launched `(h, c, w)`.
fn conv2d_1x1_stride2(a: &Args) {
    strided_conv(a, 1, 0, |n| n / 2);
}

/// The stride-2 convolutions: `k x k` taps from `(2h - pad, 2w - pad)`.
fn strided_conv(a: &Args, k: usize, pad: i64, out_size: fn(usize) -> usize) {
    with_conv(a, k, |conv, output, oc| {
        let (oh, ow) = (out_size(conv.in_h), out_size(conv.in_w));
        per_output(a, output, (oh, ow, oc), HCW, |h, w, c| {
            conv.tap_sum(c, h as i64 * 2 - pad, w as i64 * 2 - pad)
        });
    });
}

/// `relu_activation(input, output, uint size)`: `fmax(0, x)`.
pub(super) fn relu_activation(a: &Args) {
    let input = a.input::<f32>(0);
    let mut output = a.output::<f32>(1);
    let size = uint(a, 2);
    per_item(a, &mut output, size, |i| 0f32.max(input[i]));
}

/// `gelu(input, output, uint size)`: the tanh approximation.
fn gelu(a: &Args) {
    let input = a.input::<f32>(0);
    let mut output = a.output::<f32>(1);
    let size = uint(a, 2);
    per_item(a, &mut output, size, |i| {
        let x = input[i];
        let sqrt_2_over_pi = 0.797_884_6_f32;
        let x_cubed = x * x * x;
        let inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed);
        0.5 * x * (1.0 + inner.tanh())
    });
}

/// `softmax(input, output, uint size)`. Every work item computes the same
/// maximum and sum over the whole input; they are computed once here.
pub(super) fn softmax(a: &Args) {
    let input = a.input::<f32>(0);
    let mut output = a.output::<f32>(1);
    let size = uint(a, 2);
    if size.min(a.global_size()) == 0 {
        return;
    }
    let mut max_val = input[0];
    for &x in &input[1..size] {
        max_val = max_val.max(x);
    }
    let mut sum = 0f32;
    for &x in &input[..size] {
        sum += (x - max_val).exp();
    }
    per_item(a, &mut output, size, |i| (input[i] - max_val).exp() / sum);
}

/// `sigmoid(input, output, uint size)`.
fn sigmoid(a: &Args) {
    let input = a.input::<f32>(0);
    let mut output = a.output::<f32>(1);
    let size = uint(a, 2);
    per_item(a, &mut output, size, |i| 1.0 / (1.0 + (-input[i]).exp()));
}

/// `max_pool_2x2(input, output, uint input_height, uint input_width, uint channels)`.
pub(super) fn max_pool_2x2(a: &Args) {
    let input = a.input::<f32>(0);
    let mut output = a.output::<f32>(1);
    let (ih, iw, ch) = (uint(a, 2), uint(a, 3), uint(a, 4));
    per_output(a, &mut output, (ih / 2, iw / 2, ch), HWC, |h, w, c| {
        let mut max_val = f32::NEG_INFINITY;
        for kh in 0..2 {
            for kw in 0..2 {
                let idx = c * ih * iw + (h * 2 + kh) * iw + (w * 2 + kw);
                max_val = max_val.max(input[idx]);
            }
        }
        max_val
    });
}

/// `max_pool_3x3_stride2(...)`: padding 1, output `input / 2`.
fn max_pool_3x3_stride2(a: &Args) {
    let input = a.input::<f32>(0);
    let mut output = a.output::<f32>(1);
    let (ih, iw, ch) = (uint(a, 2), uint(a, 3), uint(a, 4));
    per_output(a, &mut output, (ih / 2, iw / 2, ch), HWC, |h, w, c| {
        let mut max_val = f32::NEG_INFINITY;
        for kh in 0..3 {
            for kw in 0..3 {
                let (y, x) = (h as i64 * 2 - 1 + kh, w as i64 * 2 - 1 + kw);
                if y >= 0 && y < ih as i64 && x >= 0 && x < iw as i64 {
                    let idx = c * ih * iw + y as usize * iw + x as usize;
                    max_val = max_val.max(input[idx]);
                }
            }
        }
        max_val
    });
}

/// `global_avg_pool(input, output, uint input_height, uint input_width, uint channels)`.
fn global_avg_pool(a: &Args) {
    let input = a.input::<f32>(0);
    let mut output = a.output::<f32>(1);
    let (ih, iw, ch) = (uint(a, 2), uint(a, 3), uint(a, 4));
    let plane = ih * iw;
    per_item(a, &mut output, ch, |c| {
        let mut sum = 0f32;
        for &x in &input[c * plane..(c + 1) * plane] {
            sum += x;
        }
        sum / plane as f32
    });
}

/// `global_max_pool(input, output, uint input_height, uint input_width, uint channels)`.
fn global_max_pool(a: &Args) {
    let input = a.input::<f32>(0);
    let mut output = a.output::<f32>(1);
    let (ih, iw, ch) = (uint(a, 2), uint(a, 3), uint(a, 4));
    let plane = ih * iw;
    per_item(a, &mut output, ch, |c| {
        let mut max_val = f32::NEG_INFINITY;
        for &x in &input[c * plane..(c + 1) * plane] {
            max_val = max_val.max(x);
        }
        max_val
    });
}

/// `batch_norm(input, gamma, beta, mean, variance, output, uint height,
/// uint width, uint channels, float epsilon)`.
pub(super) fn batch_norm(a: &Args) {
    let input = a.input::<f32>(0);
    let gamma = a.input::<f32>(1);
    let beta = a.input::<f32>(2);
    let mean = a.input::<f32>(3);
    let variance = a.input::<f32>(4);
    let mut output = a.output::<f32>(5);
    let (height, width, ch) = (uint(a, 6), uint(a, 7), uint(a, 8));
    let epsilon = a.float(9);
    per_output(a, &mut output, (height, width, ch), HWC, |h, w, c| {
        let idx = c * height * width + h * width + w;
        let normalized = (input[idx] - mean[c]) / (variance[c] + epsilon).sqrt();
        gamma[c] * normalized + beta[c]
    });
}

/// `layer_norm(input, gamma, beta, output, uint input_height, uint input_width,
/// uint channels, float epsilon)`: work item `(h, w)` normalises the pixel's
/// channels. The per-pixel mean and standard deviation are computed first,
/// then every output element from them.
fn layer_norm(a: &Args) {
    let input = a.input::<f32>(0);
    let gamma = a.input::<f32>(1);
    let beta = a.input::<f32>(2);
    let mut output = a.output::<f32>(3);
    let (ih, iw, ch) = (uint(a, 4), uint(a, 5), uint(a, 6));
    let epsilon = a.float(7);
    let plane = ih * iw;
    let mut stats = vec![(0f32, 0f32); plane];
    par::map(&mut stats, |p| {
        let mut mean = 0f32;
        for c in 0..ch {
            mean += input[c * plane + p];
        }
        mean /= ch as f32;
        let mut variance = 0f32;
        for c in 0..ch {
            let diff = input[c * plane + p] - mean;
            variance += diff * diff;
        }
        variance /= ch as f32;
        (mean, (variance + epsilon).sqrt())
    });
    // Launched as (h, w): every channel of a launched pixel is written.
    let (gh, gw) = (a.global_dim(0), a.global_dim(1));
    par::map_chunks(&mut output[..ch * plane], 1, |idx, o| {
        let (c, p) = (idx / plane, idx % plane);
        if p / iw < gh && p % iw < gw {
            let (mean, std) = stats[p];
            let normalized = (input[idx] - mean) / std;
            o[0] = gamma[c] * normalized + beta[c];
        }
    });
}

/// `layer_norm_1d(input, gamma, beta, output, uint size, float epsilon)`:
/// work item 0 does everything.
fn layer_norm_1d(a: &Args) {
    let input = a.input::<f32>(0);
    let gamma = a.input::<f32>(1);
    let beta = a.input::<f32>(2);
    let mut output = a.output::<f32>(3);
    let size = uint(a, 4);
    let epsilon = a.float(5);
    if a.global_size() == 0 {
        return;
    }
    let mut mean = 0f32;
    for &x in &input[..size] {
        mean += x;
    }
    mean /= size as f32;
    let mut variance = 0f32;
    for &x in &input[..size] {
        let diff = x - mean;
        variance += diff * diff;
    }
    variance /= size as f32;
    let std = (variance + epsilon).sqrt();
    for i in 0..size {
        let normalized = (input[i] - mean) / std;
        output[i] = gamma[i] * normalized + beta[i];
    }
}

/// `fully_connected(input, weights, bias, output, uint input_size, uint output_size)`:
/// `weights` is `[output][input]`.
pub(super) fn fully_connected(a: &Args) {
    let input = a.input::<f32>(0);
    let weights = a.input::<f32>(1);
    let bias = a.input::<f32>(2);
    let mut output = a.output::<f32>(3);
    let (n_in, n_out) = (uint(a, 4), uint(a, 5));
    per_item(a, &mut output, n_out, |o| {
        let mut sum = bias[o];
        for i in 0..n_in {
            sum += input[i] * weights[o * n_in + i];
        }
        sum
    });
}

/// `matrix_multiply_simd(A, B, bias, C, uint M, uint N, uint K)`: work item
/// `gid` computes row `gid % M`, column `gid / M` of `C = A B + bias`.
fn matrix_multiply_simd(a: &Args) {
    let am = a.input::<f32>(0);
    let bm = a.input::<f32>(1);
    let bias = a.input::<f32>(2);
    let mut cm = a.output::<f32>(3);
    let (m, n, k) = (uint(a, 4), uint(a, 5), uint(a, 6));
    let global = a.global_size();
    par::map_chunks(&mut cm[..m * n], 1, |idx, out| {
        let (row, col) = (idx / n, idx % n);
        if col * m + row < global {
            let mut sum = 0f32;
            for kk in 0..k {
                sum += am[row * k + kk] * bm[kk * n + col];
            }
            out[0] = sum + bias[col];
        }
    });
}

/// `add_tensors(input1, input2, output, uint size)`.
fn add_tensors(a: &Args) {
    let input1 = a.input::<f32>(0);
    let input2 = a.input::<f32>(1);
    let mut output = a.output::<f32>(2);
    let size = uint(a, 3);
    per_item(a, &mut output, size, |i| input1[i] + input2[i]);
}

/// `lcg_random(seed)`: one step of the linear congruential generator, as a
/// value in `[0, 1]` (`(float)state / (float)UINT_MAX`).
fn lcg_unit(seed: u32) -> f32 {
    let state = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    state as f32 / u32::MAX as f32
}

/// `x` dropped (0) with probability `rate`, else scaled by `1 / (1 - rate)`.
fn drop_or_scale(x: f32, random_val: f32, rate: f32) -> f32 {
    if random_val < rate {
        0.0
    } else {
        x / (1.0 - rate)
    }
}

/// `dropout(input, output, float rate, uint seed, uint training, uint size)`.
fn dropout(a: &Args) {
    let input = a.input::<f32>(0);
    let mut output = a.output::<f32>(1);
    let rate = a.float(2);
    let seed = a.scalar::<u32>(3);
    let training = a.scalar::<u32>(4);
    let size = uint(a, 5);
    per_item(a, &mut output, size, |i| {
        if training == 0 {
            input[i]
        } else {
            drop_or_scale(input[i], lcg_unit(seed.wrapping_add(i as u32)), rate)
        }
    });
}

/// `spatial_dropout(input, output, uint channels, uint height, uint width,
/// float rate, uint seed, uint training)`: one draw per channel. Note the
/// argument order: the manifest passes `dim_1_of`, `dim_2_of`, `dim_0_of` of a
/// `C x H x W` input, so `channels` receives H, `height` W and `width` C, and
/// the guard and indexing use those names as the kernel does.
fn spatial_dropout(a: &Args) {
    let input = a.input::<f32>(0);
    let mut output = a.output::<f32>(1);
    let (channels, height, width) = (uint(a, 2), uint(a, 3), uint(a, 4));
    let rate = a.float(5);
    let seed = a.scalar::<u32>(6);
    let training = a.scalar::<u32>(7);
    per_output(a, &mut output, (height, width, channels), HWC, |h, w, c| {
        let x = input[c * height * width + h * width + w];
        if training == 0 {
            x
        } else {
            drop_or_scale(x, lcg_unit(seed.wrapping_add(c as u32)), rate)
        }
    });
}

/// `upsample_2x(input, output, uint input_height, uint input_width, uint channels)`:
/// nearest neighbour; launched `(h, c, w)` over the output.
fn upsample_2x(a: &Args) {
    let input = a.input::<f32>(0);
    let mut output = a.output::<f32>(1);
    let (ih, iw, ch) = (uint(a, 2), uint(a, 3), uint(a, 4));
    per_output(a, &mut output, (ih * 2, iw * 2, ch), HCW, |h, w, c| {
        input[c * ih * iw + (h / 2) * iw + w / 2]
    });
}
