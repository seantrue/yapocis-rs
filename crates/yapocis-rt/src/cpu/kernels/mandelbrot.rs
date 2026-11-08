//! `mandelbrot.cl`: escape iteration of each point.

use crate::cpu::{par, Args};

pub const KERNELS: super::Table = &[("mandelbrot", mandelbrot)];

/// `mandelbrot(float2* q, short* output, int maxiter)`.
fn mandelbrot(a: &Args) {
    let q = a.input::<[f32; 2]>(0);
    let mut output = a.output::<i16>(1);
    let maxiter = a.int(2);
    par::map(&mut output[..a.global_size()], |gid| {
        let [qx, qy] = q[gid];
        let (mut real, mut imag) = (0f32, 0f32);
        let mut out = 0i16;
        for curiter in 0..maxiter {
            let nreal = real * real - imag * imag + qx;
            imag = 2.0 * real * imag + qy;
            real = nreal;
            if real * real + imag * imag > 4.0 {
                // The first escape; an escape at iteration 0 reads as 0
                // ("inside"), so that point keeps going, as in the OpenCL.
                out = curiter as i16;
                if curiter > 0 {
                    break;
                }
            }
        }
        out
    });
}
