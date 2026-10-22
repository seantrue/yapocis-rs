//! `np.histogram` with uniform bins, `np.linspace`, `np.interp` and `np.cumsum`,
//! as numpy 2.5 computes them (`numpy/lib/_histograms_impl.py`,
//! `numpy/_core/function_base.py`, `compiled_base.c::arr_interp`).
//!
//! For a float32 array the bin edges are float32 (`result_type` of the data and
//! its min and max), the bin index of each value is found with float32
//! arithmetic and then corrected against the edges, and `density=True` divides
//! the counts by the float64 bin widths and the total.

use crate::random::Contraction;

/// `np.linspace(start, stop, num, endpoint=True, dtype=float32)` with float32
/// scalar bounds: `arange(num) * (delta / div) + start`, the last element `stop`.
/// `np.linspace(start, stop, num)` for float64 endpoints: `arange(num) * step +
/// start`, with the last element set to `stop`.
pub fn linspace_f64(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    let div = (num - 1) as i64;
    let delta = stop - start;
    let mut y: Vec<f64> = (0..num).map(|i| i as f64).collect();
    if div > 0 {
        let step = delta / div as f64;
        if step == 0.0 {
            for v in y.iter_mut() {
                *v /= div as f64;
                *v *= delta;
            }
        } else {
            for v in y.iter_mut() {
                *v *= step;
            }
        }
    } else {
        for v in y.iter_mut() {
            *v *= delta;
        }
    }
    for v in y.iter_mut() {
        *v += start;
    }
    if num > 1 {
        y[num - 1] = stop;
    }
    y
}

pub fn linspace_f32(start: f32, stop: f32, num: usize) -> Vec<f32> {
    if num == 0 {
        return Vec::new();
    }
    let div = (num - 1) as i64;
    let delta = stop - start;
    let mut y: Vec<f32> = (0..num).map(|i| i as f32).collect();
    if div > 0 {
        let step = delta / div as f32;
        if step == 0.0 {
            for v in y.iter_mut() {
                *v /= div as f32;
                *v *= delta;
            }
        } else {
            for v in y.iter_mut() {
                *v *= step;
            }
        }
    } else {
        for v in y.iter_mut() {
            *v *= delta;
        }
    }
    for v in y.iter_mut() {
        *v += start;
    }
    if num > 1 {
        y[num - 1] = stop;
    }
    y
}

/// `np.histogram(a, bins)[0]` as counts, and the float32 bin edges, for a
/// float32 array and an integer number of bins (no `range`, no weights).
pub fn histogram_f32(a: &[f32], bins: usize) -> (Vec<i64>, Vec<f32>) {
    assert!(bins >= 1);
    let (mut first, mut last) = if a.is_empty() {
        (0.0f32, 1.0f32)
    } else {
        // numpy's min/max: NaN propagates (not expected here)
        let first = a.iter().copied().fold(f32::INFINITY, f32::min);
        let last = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (first, last)
    };
    if first == last {
        first -= 0.5;
        last += 0.5;
    }
    let edges = linspace_f32(first, last, bins + 1);
    let mut counts = vec![0i64; bins];
    let norm_denom = last - first;
    let norm_numerator = bins as f32;
    for &v in a {
        if !(v >= first && v <= last) {
            continue;
        }
        let f_index = ((v - first) / norm_denom) * norm_numerator;
        let mut index = f_index as i64; // astype(intp): truncation toward zero
        if index == bins as i64 {
            index -= 1;
        }
        if v < edges[index as usize] {
            index -= 1;
        }
        if v >= edges[index as usize + 1] && index != bins as i64 - 1 {
            index += 1;
        }
        counts[index as usize] += 1;
    }
    (counts, edges)
}

/// `np.histogram(a, bins, density=True)[0]`: `counts / diff(edges) / counts.sum()`
/// in float64.
pub fn histogram_density_f32(a: &[f32], bins: usize) -> (Vec<f64>, Vec<f32>) {
    let (counts, edges) = histogram_f32(a, bins);
    let total: i64 = counts.iter().sum();
    let total = total as f64;
    let density: Vec<f64> = counts
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            // np.diff of the float32 edges, in float32, then cast to float64
            let db = (edges[i + 1] - edges[i]) as f64;
            c as f64 / db / total
        })
        .collect();
    (density, edges)
}

/// `np.cumsum` of a float64 array (sequential).
pub fn cumsum_f64(a: &[f64]) -> Vec<f64> {
    let mut acc = 0.0f64;
    let mut out = Vec::with_capacity(a.len());
    for (i, &v) in a.iter().enumerate() {
        acc = if i == 0 { v } else { acc + v };
        out.push(acc);
    }
    out
}

/// `binary_search_with_guess` reduced to the plain bisection it falls back to.
fn search(x: f64, xp: &[f64]) -> isize {
    let n = xp.len();
    if x > xp[n - 1] {
        return n as isize;
    }
    if x < xp[0] {
        return -1;
    }
    // largest j with xp[j] <= x
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xp[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo as isize
}

/// `np.interp(x, xp, fp)` in float64 (`arr_interp`): `xp` increasing. The
/// `slope * (x - xp[j]) + fp[j]` expression follows the contraction policy of
/// the host's numpy build.
pub fn interp(x: &[f64], xp: &[f64], fp: &[f64], contraction: Contraction) -> Vec<f64> {
    let n = xp.len();
    assert!(n >= 1 && fp.len() == n);
    let lval = fp[0];
    let rval = fp[n - 1];
    if n == 1 {
        return x
            .iter()
            .map(|&v| {
                if v < xp[0] {
                    lval
                } else if v > xp[0] {
                    rval
                } else {
                    fp[0]
                }
            })
            .collect();
    }
    let slopes: Option<Vec<f64>> = if n <= x.len() {
        Some(
            (0..n - 1)
                .map(|i| (fp[i + 1] - fp[i]) / (xp[i + 1] - xp[i]))
                .collect(),
        )
    } else {
        None
    };
    let fma = |a: f64, b: f64, c: f64| match contraction {
        Contraction::Plain => a * b + c,
        Contraction::Fma => a.mul_add(b, c),
    };
    x.iter()
        .map(|&v| {
            if v.is_nan() {
                return v;
            }
            let j = search(v, xp);
            if j == -1 {
                lval
            } else if j == n as isize {
                rval
            } else if j == n as isize - 1 {
                fp[j as usize]
            } else {
                let j = j as usize;
                if xp[j] == v {
                    return fp[j];
                }
                let slope = match &slopes {
                    Some(s) => s[j],
                    None => (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]),
                };
                let mut r = fma(slope, v - xp[j], fp[j]);
                if r.is_nan() {
                    r = fma(slope, v - xp[j + 1], fp[j + 1]);
                    if r.is_nan() && fp[j] == fp[j + 1] {
                        r = fp[j];
                    }
                }
                r
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_last_is_exact_and_counts_sum() {
        let e = linspace_f32(0.1, 0.9, 257);
        assert_eq!(e.len(), 257);
        assert_eq!(e[256], 0.9);
        assert_eq!(e[0], 0.1);
        let a: Vec<f32> = (0..1000).map(|i| i as f32 / 999.0).collect();
        let (c, edges) = histogram_f32(&a, 10);
        assert_eq!(c.iter().sum::<i64>(), 1000);
        assert_eq!(edges.len(), 11);
    }

    #[test]
    fn interp_endpoints() {
        let r = interp(
            &[-1.0, 0.0, 0.5, 1.0, 2.0],
            &[0.0, 1.0],
            &[10.0, 20.0],
            Contraction::Plain,
        );
        assert_eq!(r, vec![10.0, 10.0, 15.0, 20.0, 20.0]);
    }
}
