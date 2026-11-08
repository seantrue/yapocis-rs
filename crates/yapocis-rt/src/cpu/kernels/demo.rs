//! `demo.cl`: the test kernel that adds two arrays.

use crate::cpu::{par, Args};

pub const KERNELS: super::Table = &[("sum", sum)];

fn sum(args: &Args) {
    let a = args.input::<f32>(0);
    let b = args.input::<f32>(1);
    let mut c = args.output::<f32>(2);
    par::map(&mut c[..args.global_size()], |i| a[i] + b[i]);
}
