//! Order statistics as numpy computes them: `np.quantile` (the `linear`
//! method with a Python-float `q`) and `np.median` on float32 data.

/// The k-th smallest value (`np.partition(a, k)[k]`), NaN sorting last.
pub fn kth_f32(a: &[f32], k: usize) -> f32 {
    let mut v = a.to_vec();
    let (_, value, _) = v.select_nth_unstable_by(k, |x, y| x.total_cmp(y));
    *value
}

/// `np.quantile(a, q)` for float32 `a` and a Python float `q` (`weak_q`):
/// the virtual index `(n - 1) * q`, its floor and ceiling's values, and
/// numpy's `_lerp` in float32 with the float64 gamma cast on use.
pub fn quantile_linear_f32(a: &[f32], q: f64) -> f32 {
    let n = a.len();
    assert!(n > 0, "quantile of an empty array");
    let virtual_index = (n as f64 - 1.0) * q;
    let (prev, next) = if virtual_index >= n as f64 - 1.0 {
        (n - 1, n - 1)
    } else if virtual_index < 0.0 {
        (0, 0)
    } else {
        let p = virtual_index.floor() as usize;
        (p, p + 1)
    };
    let gamma = virtual_index - virtual_index.floor();
    let lo = kth_f32(a, prev);
    let hi = kth_f32(a, next);
    let diff = hi - lo;
    if gamma >= 0.5 {
        hi - diff * ((1.0 - gamma) as f32)
    } else {
        lo + diff * (gamma as f32)
    }
}

/// `np.median(a)` for float32 `a`: the middle value, or the float32 mean of
/// the two middle ones (`(0 + a) + b` over 2).
pub fn median_f32(a: &[f32]) -> f32 {
    let n = a.len();
    assert!(n > 0, "median of an empty array");
    if n % 2 == 1 {
        kth_f32(a, (n - 1) / 2)
    } else {
        let lo = kth_f32(a, n / 2 - 1);
        let hi = kth_f32(a, n / 2);
        ((0.0f32 + lo) + hi) / 2.0f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_and_median() {
        let a = [3.0f32, 1.0, 2.0, 4.0];
        // np.quantile([3, 1, 2, 4], 0.9) == 3.7000000000000002 -> float32 lerp: 3 + 1 * 0.7 ... numpy gives 3.7
        assert_eq!(quantile_linear_f32(&a, 0.5), 2.5);
        assert_eq!(median_f32(&a), 2.5);
        assert_eq!(median_f32(&[5.0, 1.0, 3.0]), 3.0);
        assert_eq!(quantile_linear_f32(&a, 1.0), 4.0);
        assert_eq!(quantile_linear_f32(&a, 0.0), 1.0);
    }
}
