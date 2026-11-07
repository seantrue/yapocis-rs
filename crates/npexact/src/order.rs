//! Order statistics, sorts and searches as numpy computes them: `np.quantile`
//! (the `linear` method with a Python-float `q`), `np.median` and
//! `np.nanpercentile` on float32 data; `np.sort`, `np.searchsorted`,
//! `np.lexsort` and `np.unique` with its inverse.
//!
//! numpy orders floats with NaN after everything and `-0` equal to `+0`. Where
//! `np.sort` leaves the two zeros among themselves depends on its algorithm,
//! which differs by CPU (Highway on arm64, x86-simd-sort on x86-64), so
//! [`sort_f32`] matches numpy's values but not the signs of its zeros.

use std::cmp::Ordering;

/// The k-th smallest value (`np.partition(a, k)[k]`), NaN sorting last.
pub fn kth_f32(a: &[f32], k: usize) -> f32 {
    let mut v = a.to_vec();
    let (_, value, _) = v.select_nth_unstable_by(k, |x, y| x.total_cmp(y));
    *value
}

/// `np.quantile(a, q)` for float32 `a` and a Python float `q`: NumPy 2 treats
/// `q` as weak, so it becomes float32 and everything after is float32: the
/// virtual index `(n - 1) * q`, its floor, gamma, and numpy's `_lerp`. (The
/// `quantile` fixtures caught an earlier version that did this in float64.)
pub fn quantile_linear_f32(a: &[f32], q: f64) -> f32 {
    let n = a.len();
    assert!(n > 0, "quantile of an empty array");
    let last = (n - 1) as f32;
    let virtual_index = last * (q as f32);
    let (prev, next) = if virtual_index >= last {
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
        hi - diff * (1.0 - gamma)
    } else {
        lo + diff * gamma
    }
}

/// `np.quantile(a, q)` for float32 `a` with `q` kept in float64: NumPy 1, or NumPy 2 given a
/// float64 `q` (`np.float64`, an array). The virtual index `(n - 1) * q`, its floor and gamma
/// are float64, and numpy's `_lerp` runs in float32 with the gamma cast on use. Virtual
/// Impressionist's digests were recorded this way; [`quantile_linear_f32`] is NumPy 2 with a
/// Python-float `q`, and the two differ where `(n - 1) * q` rounds differently in float32.
pub fn quantile_linear_f32_f64q(a: &[f32], q: f64) -> f32 {
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

/// numpy's sort order for floats, NaN after everything (whatever its sign),
/// with `-0` put before `+0` to make the result deterministic.
fn sort_cmp_f32(x: &f32, y: &f32) -> Ordering {
    match (x.is_nan(), y.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => x.total_cmp(y),
    }
}

/// numpy's `<` for sorting floats (`npy_sort`'s `LT`): `x < y`, or `y` is NaN and `x` is not.
fn lt_f64(x: f64, y: f64) -> bool {
    x < y || (y.is_nan() && !x.is_nan())
}

fn lt_f32(x: f32, y: f32) -> bool {
    x < y || (y.is_nan() && !x.is_nan())
}

/// `np.sort(a)` of float32: ascending, NaN last. Equal to numpy's result
/// element by element, except that here every `-0` comes before every `+0`,
/// while numpy's arrangement of the zeros depends on its sort algorithm.
pub fn sort_f32(a: &[f32]) -> Vec<f32> {
    let mut v = a.to_vec();
    v.sort_unstable_by(sort_cmp_f32);
    v
}

/// `np.searchsorted(a, v, side="left")` for a float64 `a` sorted in numpy's
/// order (NaN last): the first index whose value is not below `v`.
pub fn searchsorted_left_f64(a: &[f64], v: f64) -> usize {
    a.partition_point(|&x| lt_f64(x, v))
}

/// `np.searchsorted(a, v, side="right")`: the first index whose value is above `v`.
pub fn searchsorted_right_f64(a: &[f64], v: f64) -> usize {
    a.partition_point(|&x| !lt_f64(v, x))
}

/// `np.lexsort(keys)` of float32 keys: the indices that sort by the **last** key,
/// ties broken by the one before it, and so on; stable; NaN last; `-0` and `+0`
/// equal. `None` when the keys differ in length.
pub fn lexsort_f32(keys: &[&[f32]]) -> Option<Vec<usize>> {
    let n = keys.first().map_or(0, |k| k.len());
    if keys.iter().any(|k| k.len() != n) {
        return None;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        for key in keys.iter().rev() {
            let (x, y) = (key[i], key[j]);
            if lt_f32(x, y) {
                return Ordering::Less;
            }
            if lt_f32(y, x) {
                return Ordering::Greater;
            }
        }
        Ordering::Equal
    });
    Some(idx)
}

/// `np.nanpercentile(a, qs)` for float32 `a` and a list of percentages `qs`
/// (the `linear` method): NaN left out, `q / 100` in float64, the virtual index
/// `(n - 1) * q`, and numpy's `_lerp` of the float32 neighbours with a float64
/// gamma (the list promotes the result to float64). NaN for every `q` when
/// nothing but NaN is left.
pub fn nanpercentile_linear_f32(a: &[f32], qs: &[f64]) -> Vec<f64> {
    let mut v: Vec<f32> = a.iter().copied().filter(|x| !x.is_nan()).collect();
    let n = v.len();
    if n == 0 {
        return vec![f64::NAN; qs.len()];
    }
    v.sort_unstable_by(sort_cmp_f32);
    qs.iter()
        .map(|&q| {
            let q = q / 100.0;
            let virtual_index = (n as f64 - 1.0) * q;
            let (prev, next) = if virtual_index >= n as f64 - 1.0 {
                (n - 1, n - 1)
            } else if virtual_index < 0.0 || virtual_index.is_nan() {
                (0, 0)
            } else {
                let p = virtual_index.floor() as usize;
                (p, p + 1)
            };
            let gamma = virtual_index - virtual_index.floor();
            let lo = v.get(prev).copied().unwrap_or(f32::NAN);
            let hi = v.get(next).copied().unwrap_or(f32::NAN);
            let diff = f64::from(hi - lo);
            if gamma >= 0.5 {
                f64::from(hi) - diff * (1.0 - gamma)
            } else {
                f64::from(lo) + diff * gamma
            }
        })
        .collect()
}

/// `np.unique(a, return_inverse=True)` of int64: the distinct values ascending,
/// and for each element the index of its value among them.
pub fn unique_inverse_i64(a: &[i64]) -> (Vec<i64>, Vec<usize>) {
    let (Some(&lo), Some(&hi)) = (a.iter().min(), a.iter().max()) else {
        return (Vec::new(), Vec::new());
    };
    let span = (hi as i128 - lo as i128) as u128;
    if span < 4 * a.len() as u128 + 64 {
        // dense labels: a rank table over the range instead of a sort
        let span = span as usize + 1;
        let mut rank = vec![usize::MAX; span];
        for &v in a {
            rank[(v as i128 - lo as i128) as usize] = 0;
        }
        let mut values = Vec::new();
        for (k, r) in rank.iter_mut().enumerate() {
            if *r == 0 {
                *r = values.len();
                values.push((lo as i128 + k as i128) as i64);
            }
        }
        let inverse = a
            .iter()
            .map(|&v| rank[(v as i128 - lo as i128) as usize])
            .collect();
        return (values, inverse);
    }
    let mut values = a.to_vec();
    values.sort_unstable();
    values.dedup();
    let inverse = a
        .iter()
        .map(|v| values.binary_search(v).unwrap_or(0))
        .collect();
    (values, inverse)
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

    #[test]
    fn sorts_and_searches() {
        let s = sort_f32(&[0.0, f32::NAN, -0.0, -1.0]);
        assert_eq!(s[0], -1.0);
        assert!(s[1].is_sign_negative() && s[2] == 0.0 && s[2].is_sign_positive() && s[3].is_nan());
        let a = [1.0, 2.0, 2.0, f64::NAN];
        assert_eq!(searchsorted_left_f64(&a, 2.0), 1);
        assert_eq!(searchsorted_right_f64(&a, 2.0), 3);
        assert_eq!(searchsorted_left_f64(&a, f64::NAN), 3);
        assert_eq!(searchsorted_right_f64(&a, f64::NAN), 4);
        // np.lexsort(([1, 0, 1], [0, 0, 0])) == [1, 0, 2]
        assert_eq!(
            lexsort_f32(&[&[1.0, 0.0, 1.0], &[0.0, 0.0, 0.0]]),
            Some(vec![1, 0, 2])
        );
        assert_eq!(lexsort_f32(&[&[1.0], &[0.0, 0.0]]), None);
        assert_eq!(
            unique_inverse_i64(&[5, 5, -1, 3]),
            (vec![-1, 3, 5], vec![2, 2, 0, 1])
        );
        assert_eq!(
            unique_inverse_i64(&[i64::MAX, i64::MIN]),
            (vec![i64::MIN, i64::MAX], vec![1, 0])
        );
        assert!(nanpercentile_linear_f32(&[f32::NAN], &[50.0])[0].is_nan());
    }
}

#[cfg(test)]
mod f64q_tests {
    use super::*;

    /// Where `(n - 1) * q` rounds differently in float32 and float64, the two quantiles take
    /// different neighbours or gammas; on exact data they agree.
    #[test]
    fn the_float64_q_quantile_differs_only_where_the_index_rounds_differently() {
        let a: Vec<f32> = (0..11).map(|i| i as f32).collect();
        assert_eq!(quantile_linear_f32_f64q(&a, 0.25), 2.5);
        assert_eq!(quantile_linear_f32(&a, 0.25), 2.5);
        let mut differ = 0;
        for n in 2..400usize {
            let a: Vec<f32> = (0..n)
                .map(|i| ((i * 7919) % 1009) as f32 / 1009.0)
                .collect();
            for k in 1..100 {
                let q = f64::from(k) / 100.0;
                let (strong, weak) = (quantile_linear_f32_f64q(&a, q), quantile_linear_f32(&a, q));
                if strong.to_bits() != weak.to_bits() {
                    differ += 1;
                }
                // the float64 index, as the function computes it
                let vi = (n as f64 - 1.0) * q;
                let (lo, hi) = (
                    kth_f32(&a, vi.floor() as usize),
                    kth_f32(&a, (vi.floor() as usize + 1).min(n - 1)),
                );
                let g = vi - vi.floor();
                let want = if g >= 0.5 {
                    hi - (hi - lo) * ((1.0 - g) as f32)
                } else {
                    lo + (hi - lo) * (g as f32)
                };
                assert_eq!(strong.to_bits(), want.to_bits(), "n {n} q {q}");
            }
        }
        assert!(differ > 0, "the two quantiles never differed");
    }
}
