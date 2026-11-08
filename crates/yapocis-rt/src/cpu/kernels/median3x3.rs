//! `median3x3.cl`: 3x3 median filter on a 2-D image addressed `x * height + y`.

use crate::cpu::{par, Args};

pub const KERNELS: super::Table = &[("median3x3", median3x3)];

/// `median9`: the first five passes of a selection sort over nine values
/// (swap `a[q]` into `a[p]` when `a[q] < a[p]`), then `a[4]`. A NaN never
/// compares less, so it stays where it is.
fn median9(mut a: [f32; 9]) -> f32 {
    for p in 0..5 {
        for q in p + 1..9 {
            if a[q] < a[p] {
                a.swap(p, q);
            }
        }
    }
    a[4]
}

/// `median3x3(int width, int height, float* a, float* ret)`.
///
/// `width` is the array's total size (`size_of a`). Pixels with flat index
/// in `height + 1 .. width - (height + 1)` take the median of their 3x3
/// neighbourhood in flat-index terms (so the first and last rows of each
/// column wrap into the neighbouring column); the rest are copied.
fn median3x3(args: &Args) {
    let size = args.int(0) as usize;
    let h = args.int(1) as usize;
    let a = args.input::<f32>(2);
    let mut ret = args.output::<f32>(3);
    let left = h + 1;
    let right = size.wrapping_sub(left);
    par::map(&mut ret[..args.global_size()], |i| {
        if i >= left && i < right {
            median9([
                a[i - h - 1],
                a[i - h],
                a[i - h + 1],
                a[i - 1],
                a[i],
                a[i + 1],
                a[i + h - 1],
                a[i + h],
                a[i + h + 1],
            ])
        } else {
            a[i]
        }
    });
}
