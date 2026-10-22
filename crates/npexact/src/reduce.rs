//! numpy's floating-point reductions, in numpy's summation order.
//!
//! `np.sum`, `np.mean`, `np.var` and `np.std` over a contiguous float array go
//! through `@TYPE@_pairwise_sum` (`numpy/_core/src/umath/loops_utils.h.src`):
//! fewer than 8 elements are added sequentially starting from -0.0, blocks of up
//! to 128 use eight interleaved accumulators, and longer arrays split in two
//! (the first half rounded down to a multiple of 8) recursively. The reduction
//! itself starts from `add`'s identity, `+0.0`, and adds that pairwise sum, so
//! `np.sum([-0.0])` is `+0.0` and an empty sum is `0.0`. `mean` divides the sum
//! by the count in the array's dtype, and `var` is the pairwise sum of the
//! squared deviations from that mean (`numpy/_core/_methods.py`).

/// `FLOAT_pairwise_sum` over a contiguous slice.
pub fn pairwise_sum_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0f32;
        for &v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for (j, acc) in r.iter_mut().enumerate() {
                *acc += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum_f32(&a[..n2]) + pairwise_sum_f32(&a[n2..])
    }
}

/// `DOUBLE_pairwise_sum` over a contiguous slice.
pub fn pairwise_sum_f64(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0f64;
        for &v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for (j, acc) in r.iter_mut().enumerate() {
                *acc += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum_f64(&a[..n2]) + pairwise_sum_f64(&a[n2..])
    }
}

/// `np.sum(a)` of a contiguous float32 array (dtype float32).
pub fn sum_f32(a: &[f32]) -> f32 {
    0.0 + pairwise_sum_f32(a)
}

/// `np.sum(a)` of a contiguous float64 array.
pub fn sum_f64(a: &[f64]) -> f64 {
    0.0 + pairwise_sum_f64(a)
}

/// `np.mean(a)` of a float32 array: the pairwise sum divided by the count in float32.
pub fn mean_f32(a: &[f32]) -> f32 {
    sum_f32(a) / a.len() as f32
}

/// `np.mean(a)` of a float64 array.
pub fn mean_f64(a: &[f64]) -> f64 {
    sum_f64(a) / a.len() as f64
}

/// `np.var(a, ddof)` of a float32 array (`_methods._var`): deviations from the
/// float32 mean, squared, pairwise-summed, divided by `max(n - ddof, 0)`.
pub fn var_f32(a: &[f32], ddof: usize) -> f32 {
    let mean = mean_f32(a);
    let sq: Vec<f32> = a
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .collect();
    let rcount = a.len().saturating_sub(ddof) as f32;
    sum_f32(&sq) / rcount
}

/// `np.var(a, ddof)` of a float64 array.
pub fn var_f64(a: &[f64], ddof: usize) -> f64 {
    let mean = mean_f64(a);
    let sq: Vec<f64> = a
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .collect();
    let rcount = a.len().saturating_sub(ddof) as f64;
    sum_f64(&sq) / rcount
}

/// `np.std(a, ddof)` of a float32 array: `sqrt(var)`.
pub fn std_f32(a: &[f32], ddof: usize) -> f32 {
    var_f32(a, ddof).sqrt()
}

/// `np.std(a, ddof)` of a float64 array.
pub fn std_f64(a: &[f64], ddof: usize) -> f64 {
    var_f64(a, ddof).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_zero_in_the_pairwise_sum_but_not_the_reduction() {
        assert_eq!(
            pairwise_sum_f32(&[-0.0, -0.0]).to_bits(),
            (-0.0f32).to_bits()
        );
        assert_eq!(sum_f32(&[-0.0, -0.0]).to_bits(), 0.0f32.to_bits());
        assert_eq!(sum_f64(&[]).to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn big_and_ones_follow_the_block_order() {
        // 2^60 then 999 ones then -2^60: sequential summation loses every 1,
        // the pairwise order keeps 640 of them (spike S2b, numpy 2.5.3)
        let big = 2f64.powi(60);
        let mut a = vec![1.0f64; 1000];
        a[0] = big;
        a[999] = -big;
        assert_eq!(sum_f64(&a), 640.0);
    }
}

/// `np.add.reduceat(a, starts, dtype=float64)` on float32 (or float64) input:
/// each segment is its first element plus the pairwise sum of the rest (the
/// reduceat loop copies the first element and reduces the remainder; unlike
/// `np.sum`, no identity is used). Segments run to the next start or the end.
pub fn reduceat_add_f64(a: &[f64], starts: &[usize]) -> Vec<f64> {
    starts
        .iter()
        .enumerate()
        .map(|(k, &s)| {
            let e = if k + 1 < starts.len() {
                starts[k + 1]
            } else {
                a.len()
            };
            let seg = &a[s..e];
            if seg.len() <= 1 {
                seg.first().copied().unwrap_or(0.0)
            } else {
                seg[0] + pairwise_sum_f64(&seg[1..])
            }
        })
        .collect()
}

#[cfg(test)]
mod reduceat_tests {
    use super::reduceat_add_f64;

    #[test]
    fn first_plus_pairwise_of_the_rest() {
        // np.add.reduceat(np.float32([1e8, 1.0, 1.0, -1e8]), [0, 2], dtype=np.float64)
        // -> [1e8 + (0 + 1.0) = 100000001.0, 1.0 + (0 + -1e8) = -99999999.0]
        let a = [1e8f64, 1.0, 1.0, -1e8];
        assert_eq!(
            reduceat_add_f64(&a, &[0, 2]),
            vec![100000001.0, -99999999.0]
        );
        assert_eq!(reduceat_add_f64(&[3.0], &[0]), vec![3.0]);
    }
}
