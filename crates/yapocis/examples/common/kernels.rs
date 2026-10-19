//! Port of `yapocis.lib.convolve_kernels.generate_kernel` for the examples
//! (the weight generators behind the bundled `convolves` interface).
#![allow(dead_code)]

fn normalized(w: Vec<f64>) -> Vec<f64> {
    let s: f64 = w.iter().sum();
    w.into_iter().map(|v| v / s).collect()
}

fn cosine_window(size: usize, f: impl Fn(f64) -> f64) -> Vec<f64> {
    let m = (size - 1) as f64;
    normalized((0..size).map(|n| f(n as f64 / m)).collect())
}

pub fn generate_kernel(kernel_type: &str, size: usize) -> Vec<f32> {
    use std::f64::consts::PI;
    let w: Vec<f64> = match kernel_type {
        "hamming" => cosine_window(size, |x| 0.54 - 0.46 * (2.0 * PI * x).cos()),
        "hann" => cosine_window(size, |x| 0.5 * (1.0 - (2.0 * PI * x).cos())),
        "blackman" => cosine_window(size, |x| {
            0.42 - 0.5 * (2.0 * PI * x).cos() + 0.08 * (4.0 * PI * x).cos()
        }),
        "bartlett" | "triangle" => {
            let half = (size - 1) as f64 / 2.0;
            normalized(
                (0..size)
                    .map(|n| 1.0 - ((n as f64 - half) / half).abs())
                    .collect(),
            )
        }
        "box" => vec![1.0 / size as f64; size],
        "binomial" => {
            let mut w = vec![1.0f64];
            for _ in 0..size - 1 {
                let mut next = vec![0.0; w.len() + 1];
                for (i, v) in w.iter().enumerate() {
                    next[i] += v;
                    next[i + 1] += v;
                }
                w = next;
            }
            normalized(w)
        }
        "deriv1" => match size {
            5 => [1.0, -8.0, 0.0, 8.0, -1.0]
                .iter()
                .map(|v| v / 12.0)
                .collect(),
            7 => [-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0]
                .iter()
                .map(|v| v / 60.0)
                .collect(),
            _ => vec![-0.5, 0.0, 0.5],
        },
        "deriv2" => match size {
            5 => [-1.0, 16.0, -30.0, 16.0, -1.0]
                .iter()
                .map(|v| v / 12.0)
                .collect(),
            7 => [2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0]
                .iter()
                .map(|v| v / 180.0)
                .collect(),
            _ => vec![1.0, -2.0, 1.0],
        },
        "sobel" | "prewitt" => {
            if size == 3 {
                vec![-1.0, 0.0, 1.0]
            } else {
                generate_kernel("deriv1", size)
                    .iter()
                    .map(|v| *v as f64)
                    .collect()
            }
        }
        "sharpen" => match size {
            5 => vec![-1.0, -1.0, 5.0, -1.0, -1.0],
            _ => vec![-1.0, 3.0, -1.0],
        },
        "unsharp" => {
            let mut k = vec![-1.0 / size as f64; size];
            k[size / 2] += 2.0;
            k
        }
        "dog" => {
            let n = match size {
                5 | 7 | 9 | 11 => size,
                _ => 5,
            };
            let c = (n - 1) as f64;
            (0..n)
                .map(|i| if i == n / 2 { 1.0 } else { -1.0 / c })
                .collect()
        }
        other => panic!("Unknown kernel type: {other}"),
    };
    w.into_iter().map(|v| v as f32).collect()
}
