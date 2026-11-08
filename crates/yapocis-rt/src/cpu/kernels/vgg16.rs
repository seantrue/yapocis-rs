//! `vgg16.cl`: the VGG16 layers. All but `matrix_multiply_simd` compute what
//! `cnn.cl`'s kernels of the same name do (the 3x3 convolutions visit their
//! taps in the same order, and `max` matches `fmax` for numbers), so they
//! share those ports.

use super::cnn;
use crate::cpu::{par, Args};

pub const KERNELS: super::Table = &[
    ("conv2d_3x3", cnn::conv2d_3x3),
    ("conv2d_3x3_optimized", cnn::conv2d_3x3),
    ("relu_activation", cnn::relu_activation),
    ("softmax", cnn::softmax),
    ("max_pool_2x2", cnn::max_pool_2x2),
    ("batch_norm", cnn::batch_norm),
    ("fully_connected", cnn::fully_connected),
    ("matrix_multiply_simd", matrix_multiply_simd),
];

/// `matrix_multiply_simd(A, B, bias, C, uint M, uint N, uint K)`: work item
/// `(row, col)` of a 2-D launch computes `C[row][col] = A B + bias`. The
/// interface's dispatch size is 1-D (`sizeof C`), so only column 0 is
/// computed unless the caller passes a 2-D global size.
fn matrix_multiply_simd(a: &Args) {
    let am = a.input::<f32>(0);
    let bm = a.input::<f32>(1);
    let bias = a.input::<f32>(2);
    let mut cm = a.output::<f32>(3);
    let m = a.scalar::<u32>(4) as usize;
    let n = a.scalar::<u32>(5) as usize;
    let k = a.scalar::<u32>(6) as usize;
    let (rows, cols) = (a.global_dim(0), a.global_dim(1));
    par::map_chunks(&mut cm[..m * n], 1, |idx, out| {
        let (row, col) = (idx / n, idx % n);
        if row < rows && col < cols {
            let mut sum = 0f32;
            for kk in 0..k {
                sum += am[row * k + kk] * bm[kk * n + col];
            }
            out[0] = sum + bias[col];
        }
    });
}
