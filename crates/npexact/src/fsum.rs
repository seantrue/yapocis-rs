//! Python's `math.fsum`: Shewchuk's exactly rounded summation (`mathmodule.c`
//! `math_fsum`), for sums that must not depend on summation order.

/// `math.fsum(values)` for finite values: the correctly rounded sum. A
/// non-finite input gives the IEEE result of adding it (`inf`, `-inf` or `nan`),
/// as Python returns for one infinity or a NaN.
pub fn fsum(values: &[f64]) -> f64 {
    if values.iter().any(|v| !v.is_finite()) {
        return values.iter().sum();
    }
    let mut partials: Vec<f64> = Vec::new();
    for &v in values {
        let mut x = v;
        let mut i = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        partials.truncate(i);
        partials.push(x);
    }
    // sum the partials from the end, rounding once
    let mut n = partials.len();
    if n == 0 {
        return 0.0;
    }
    n -= 1;
    let mut hi = partials[n];
    let mut lo = 0.0;
    while n > 0 {
        let x = hi;
        n -= 1;
        let y = partials[n];
        hi = x + y;
        let yr = hi - x;
        lo = y - yr;
        if lo != 0.0 {
            break;
        }
    }
    if n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0)) {
        let y = lo * 2.0;
        let x = hi + y;
        let yr = x - hi;
        if y == yr {
            hi = x;
        }
    }
    hi
}

#[cfg(test)]
mod tests {
    use super::fsum;

    #[test]
    fn exact_where_naive_sums_are_not() {
        // Python: math.fsum([1e100, 1.0, -1e100, 1e-100, 1e50, -1.0, -1e50]) == 1e-100
        assert_eq!(
            fsum(&[1e100, 1.0, -1e100, 1e-100, 1e50, -1.0, -1e50]),
            1e-100
        );
        // math.fsum([0.1] * 10) == 1.0 while sum gives 0.9999999999999999
        assert_eq!(fsum(&[0.1; 10]), 1.0);
        assert_eq!(fsum(&[]), 0.0);
        assert_eq!(fsum(&[1.0, 1e16, 1.0]), 10000000000000002.0);
    }
}
