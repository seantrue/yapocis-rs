//! `fft.cl`: radix-2 FFTs (single and batched), windows and power spectra.
//!
//! The FFT kernels run one work group through a shared-memory Cooley-Tukey
//! FFT. The port runs that group's stages one after another; within a stage
//! the butterflies touch disjoint pairs, so their order does not matter, and
//! the twiddles and butterflies are computed in `f32` exactly as the kernel
//! writes them. When the work group is not the shape that path needs, the
//! kernel falls back to a direct DFT per work item, and so does the port.
//! The program is built with `-cl-opt-disable`, so there is no fast math to
//! imitate.

use crate::cpu::{par, Args};

pub const KERNELS: super::Table = &[
    ("fft1d", fft1d),
    ("fft1d_batch", fft1d_batch),
    ("window", window),
    ("power_spectrum", power_spectrum),
];

const M_PI: f32 = std::f32::consts::PI;
const M_2PI: f32 = 2.0 * std::f32::consts::PI;

type Complex = [f32; 2];

fn cmul(a: Complex, b: Complex) -> Complex {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

fn cadd(a: Complex, b: Complex) -> Complex {
    [a[0] + b[0], a[1] + b[1]]
}

fn csub(a: Complex, b: Complex) -> Complex {
    [a[0] - b[0], a[1] - b[1]]
}

fn bit_reverse_int(mut x: usize, log2n: u32) -> usize {
    let mut result = 0;
    for _ in 0..log2n {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// `twiddle_factor(k, N, direction)`: `angle = -direction * M_2PI * k / N`.
fn twiddle_factor(k: usize, n: usize, direction: i32) -> Complex {
    let angle = (-direction) as f32 * M_2PI * k as f32 / n as f32;
    [angle.cos(), angle.sin()]
}

/// `log2(n)` the way the kernels compute it: shift right until 1.
fn log2(n: usize) -> u32 {
    let (mut temp, mut log2n) = (n, 0);
    while temp > 1 {
        temp >>= 1;
        log2n += 1;
    }
    log2n
}

/// One work group's radix-2 FFT of `input` (length `n`, a power of two),
/// run by its first `lanes` work items. Returns the group's `shared_data`.
fn radix2_group(input: &[Complex], n: usize, lanes: usize, direction: i32) -> Vec<Complex> {
    let log2n = log2(n);
    let mut shared = vec![[0f32; 2]; n];
    for lid in 0..lanes {
        shared[lid] = input[bit_reverse_int(lid, log2n)];
    }
    // barrier
    for stage in 0..log2n {
        let m = 1usize << (stage + 1);
        let m2 = m >> 1;
        for lid in 0..lanes {
            let k = lid % m2;
            let j = (lid / m) * m + k;
            // Only the lower half of each block runs the butterfly.
            if lid % m < m2 && j + m2 < n {
                let w = twiddle_factor(k, m, direction);
                let t = cmul(w, shared[j + m2]);
                let u = shared[j];
                shared[j] = cadd(u, t);
                shared[j + m2] = csub(u, t);
            }
        }
        // barrier
    }
    shared
}

/// Divide by `n` for the inverse transform.
fn normalize(v: Complex, n: usize, direction: i32) -> Complex {
    if direction == -1 {
        [v[0] / n as f32, v[1] / n as f32]
    } else {
        v
    }
}

/// The DFT fallback's output `index`: `angle = -direction * M_2PI * index * k / n`.
fn dft(input: &[Complex], n: usize, index: usize, direction: i32) -> Complex {
    let mut sum = [0f32; 2];
    for (k, &x) in input.iter().enumerate().take(n) {
        let angle = (-direction) as f32 * M_2PI * index as f32 * k as f32 / n as f32;
        sum = cadd(sum, cmul(x, [angle.cos(), angle.sin()]));
    }
    normalize(sum, n, direction)
}

/// `fft1d(float2* input, int n, float2* output, int direction)`.
///
/// The radix-2 path needs exactly one work group of size `n` (`n` a power of
/// two up to 1024). A launch without a local size leaves the group size to
/// the OpenCL driver; here it then takes the DFT path.
fn fft1d(a: &Args) {
    let input = a.input::<Complex>(0);
    let n = a.int(1) as usize;
    let mut output = a.output::<Complex>(2);
    let direction = a.int(3);
    let local_size = a.local().map(|l| l[0]);
    let items = a.global_size().min(n); // work items with gid < n
    if n <= 1024 && n.is_power_of_two() && local_size == Some(n) {
        let shared = radix2_group(&input, n, n, direction);
        for lid in 0..items {
            output[lid] = normalize(shared[lid], n, direction);
        }
    } else {
        par::map(&mut output[..items], |gid| dft(&input, n, gid, direction));
    }
}

/// `fft1d_batch(float2* input, int batch_size, int fft_size, int direction,
/// float2* output)`: work group `g` transforms row `g` of a
/// `[batch_size][fft_size]` array. Groups are independent and run in
/// parallel. Without a local size the group size is taken as `fft_size`.
fn fft1d_batch(a: &Args) {
    let input = a.input::<Complex>(0);
    let batch_size = a.int(1) as usize;
    let n = a.int(2) as usize;
    let direction = a.int(3);
    let mut output = a.output::<Complex>(4);
    let group_size = a.local().map_or(n, |l| l[0]).max(1);
    let groups = (a.global_size() / group_size).min(batch_size);
    let lanes = group_size.min(n); // work items with lid < n
    par::map_chunks(&mut output[..groups * n], n, |batch, row| {
        let row_in = &input[batch * n..(batch + 1) * n];
        if n <= 256 && n.is_power_of_two() {
            let shared = radix2_group(row_in, n, lanes, direction);
            for lid in 0..lanes {
                row[lid] = normalize(shared[lid], n, direction);
            }
        } else {
            for (lid, out) in row.iter_mut().enumerate().take(lanes) {
                *out = dft(row_in, n, lid, direction);
            }
        }
    });
}

/// `window(float* input, int window_type, float* output)`: 0 rectangular,
/// 1 Hann, 2 Hamming, 3 Blackman, over the global size.
fn window(a: &Args) {
    let input = a.input::<f32>(0);
    let window_type = a.int(1);
    let mut output = a.output::<f32>(2);
    let n = a.global_size();
    par::map(&mut output[..n], |gid| {
        let pos = gid as f32 / (n - 1) as f32;
        let w = match window_type {
            1 => 0.5 * (1.0 - (M_2PI * pos).cos()),
            2 => 0.54 - 0.46 * (M_2PI * pos).cos(),
            3 => 0.42 - 0.5 * (M_2PI * pos).cos() + 0.08 * (4.0 * M_PI * pos).cos(),
            _ => 1.0,
        };
        input[gid] * w
    });
}

/// `power_spectrum(float2* fft_output, float* power)`: `|z|^2`.
fn power_spectrum(a: &Args) {
    let z = a.input::<Complex>(0);
    let mut power = a.output::<f32>(1);
    par::map(&mut power[..a.global_size()], |i| {
        z[i][0] * z[i][0] + z[i][1] * z[i][1]
    });
}
