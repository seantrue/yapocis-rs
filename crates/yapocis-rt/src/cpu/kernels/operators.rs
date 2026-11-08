//! `operators.cl`: pixel-wise arithmetic on two images.

use crate::cpu::{par, Args};

pub const KERNELS: super::Table = &[("add", add), ("sub", sub), ("mul", mul), ("divop", divop)];

/// `output[i] = op(input1[i], input2[i])`, the shape all four kernels share.
fn pixelwise(a: &Args, op: impl Fn(f32, f32) -> f32 + Sync) {
    let input1 = a.input::<f32>(0);
    let input2 = a.input::<f32>(1);
    let mut output = a.output::<f32>(2);
    par::map(&mut output[..a.global_size()], |i| op(input1[i], input2[i]));
}

fn add(a: &Args) {
    pixelwise(a, |x, y| x + y)
}

fn sub(a: &Args) {
    pixelwise(a, |x, y| x - y)
}

fn mul(a: &Args) {
    pixelwise(a, |x, y| x * y)
}

fn divop(a: &Args) {
    pixelwise(a, |x, y| x / y)
}
