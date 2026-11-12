//! The convolution families: `convolve.mako` (one kernel and its `_fast`
//! form per variant), `convolves.mako` (many named 1-D kernels) and
//! `convolvesep.mako` (separable Gaussians: a main kernel and `_h`/`_v`).
//!
//! Their weights are rendered into the OpenCL text as unrolled lines such as
//! `sum = sum + a[i+(-1)] * 0.3333333333;`, so the CPU backend reads them
//! back from the source it is given, bundled or generated at run time by
//! `yapocis::lib::convolve`. The weights are `double` literals in OpenCL C,
//! so each step `sum = sum + a[k] * w` is computed in `f64` and rounded to
//! `float` when it is stored, as the C does.

use crate::cpu::{par, Args};

/// The kernels in an OpenCL source, in order, with the weights of each: the
/// literals on its first run of lines that index `a[i+(`.
fn parse(source: &str) -> Result<Vec<(String, Vec<f64>)>, String> {
    let mut kernels = Vec::new();
    for chunk in source.split("__kernel").skip(1) {
        let Some(void) = chunk.find("void ") else {
            continue;
        };
        let rest = &chunk[void + 5..];
        let name = rest[..rest.find('(').ok_or("kernel without '('")?].trim();
        let mut weights = Vec::new();
        for line in chunk.lines() {
            if !line.contains("a[i+(") {
                if weights.is_empty() {
                    continue;
                }
                break; // the end of the first run of weight lines
            }
            // `... a[i+(off)] * <weight>;` or `... a[i+(off)] * <weight> : 0.0f;`
            let after = &line[line
                .rfind("] * ")
                .ok_or_else(|| format!("{name}: {line}"))?
                + 4..];
            let literal = after.split([';', ' ']).next().unwrap_or("");
            let w: f64 = literal
                .parse()
                .map_err(|e| format!("{name}: weight {literal:?}: {e}"))?;
            weights.push(w);
        }
        kernels.push((name.to_string(), weights));
    }
    if kernels.is_empty() {
        return Err("no kernels in the OpenCL source".into());
    }
    Ok(kernels)
}

/// `sum + x * w` as OpenCL C computes it with a `double` literal `w`.
fn step(sum: f32, x: f32, w: f64) -> f32 {
    (sum as f64 + x as f64 * w) as f32
}

// ---------------------------------------------------------------------------
// convolve.mako and convolves.mako
// ---------------------------------------------------------------------------

/// The kernels of a `convolve` variant or of `convolves`.
pub fn one_dimensional(source: &str) -> Result<Vec<(String, super::Kernel)>, String> {
    let parsed = parse(source)?;
    let weights_of = |name: &str| {
        parsed
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, w)| w.clone())
            .ok_or_else(|| format!("{name}: no weights"))
    };
    let mut out: Vec<(String, super::Kernel)> = Vec::new();
    for (name, weights) in &parsed {
        if let Some(base) = name.strip_suffix("_fast") {
            // The _fast body repeats its base kernel's weights.
            let w = weights_of(base)?;
            out.push((name.clone(), Box::new(move |a: &Args| convolve_fast(a, &w))));
        } else {
            let w = weights.clone();
            out.push((name.clone(), Box::new(move |a: &Args| convolve(a, &w))));
        }
    }
    Ok(out)
}

/// `NAME(int width, float* a, float* ret)`: a centred 1-D convolution that
/// leaves zeros where the window would leave the array. Note the strict
/// `i > left`, which also zeroes `ret[left]`.
fn convolve(args: &Args, weights: &[f64]) {
    let width = args.int(0) as i64;
    let a = args.input::<f32>(1);
    let mut ret = args.output::<f32>(2);
    let left = (weights.len() / 2) as i64;
    let right = width - left;
    par::map(&mut ret[..args.global_size()], |i| {
        let i = i as i64;
        let mut sum = 0f32;
        if i > left && i < right {
            for (k, &w) in weights.iter().enumerate() {
                sum = step(sum, a[(i + k as i64 - left) as usize], w);
            }
        }
        sum
    });
}

/// `NAME_fast(int width, float* a, float* ret, int chunk)`: work item `g`
/// computes `ret[g*chunk .. g*chunk+chunk-1]`, each pixel exactly as `NAME`
/// does, and stops at `width`; a `chunk` below 1 counts as 1 (Yapocis3 #80).
/// Any global size of at least `ceil(width / chunk)` covers every pixel.
fn convolve_fast(args: &Args, weights: &[f64]) {
    let width = args.int(0) as i64;
    let a = args.input::<f32>(1);
    let mut ret = args.output::<f32>(2);
    let n = (args.int(3) as i64).max(1);
    let left = (weights.len() / 2) as i64;
    let right = width - left;
    for g in 0..args.global_size() as i64 {
        for j in 0..n {
            let i = g * n + j;
            if i >= width {
                break;
            }
            let mut sum = 0f32;
            if i > left && i < right {
                for (k, &w) in weights.iter().enumerate() {
                    sum = step(sum, a[(i + k as i64 - left) as usize], w);
                }
            }
            ret[i as usize] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// convolvesep.mako
// ---------------------------------------------------------------------------

/// The kernels of `convolvesep` (bundled or generated): `NAME`, `NAME_h`, `NAME_v`.
pub fn separable(source: &str) -> Result<Vec<(String, super::Kernel)>, String> {
    let mut out: Vec<(String, super::Kernel)> = Vec::new();
    for (name, weights) in parse(source)? {
        let kernel: super::Kernel = if name.ends_with("_h") {
            Box::new(move |a: &Args| separable_pass(a, &weights, Pass::Horizontal))
        } else if name.ends_with("_v") {
            Box::new(move |a: &Args| separable_pass(a, &weights, Pass::Vertical))
        } else {
            Box::new(move |a: &Args| separable_main(a, &weights))
        };
        out.push((name, kernel));
    }
    Ok(out)
}

/// `NAME(int vertical, int width, int height, float* a, float* ret)`.
///
/// The image is addressed `i = x * height + y`. With `vertical == 0` the
/// taps step by `height` (along x) and the test is on the flat index:
/// `margin * height < i < n - margin * height`; with `vertical != 0` they
/// step by 1 and the test is `margin < i < n - margin`.
fn separable_main(args: &Args, weights: &[f64]) {
    let vertical = args.int(0) != 0;
    let width = args.int(1) as i64;
    let height = args.int(2) as i64;
    let a = args.input::<f32>(3);
    let mut ret = args.output::<f32>(4);
    let n = width * height;
    let margin = (weights.len() / 2) as i64;
    let (stride, low, high) = if vertical {
        (1, margin, n - margin)
    } else {
        (height, margin * height, n - margin * height)
    };
    par::map(&mut ret[..args.global_size()], |i| {
        let i = i as i64;
        let mut sum = 0f32;
        if low < i && i < high {
            for (k, &w) in weights.iter().enumerate() {
                let off = (k as i64 - margin) * stride;
                sum = step(sum, a[(i + off) as usize], w);
            }
        }
        sum
    });
}

enum Pass {
    Horizontal,
    Vertical,
}

/// `NAME_h` / `NAME_v(int width, int height, float* a, float* ret)`: the taps
/// step by `height` (`_h`) or 1 (`_v`), and only pixels at least `margin`
/// from every edge are convolved; the rest are 0.
fn separable_pass(args: &Args, weights: &[f64], pass: Pass) {
    let width = args.int(0) as i64;
    let height = args.int(1) as i64;
    let a = args.input::<f32>(2);
    let mut ret = args.output::<f32>(3);
    let margin = (weights.len() / 2) as i64;
    let stride = match pass {
        Pass::Horizontal => height,
        Pass::Vertical => 1,
    };
    par::map(&mut ret[..args.global_size()], |i| {
        let i = i as i64;
        let (x, y) = (i / height, i % height);
        let inside = x >= margin && x < width - margin && y >= margin && y < height - margin;
        let mut sum = 0f32;
        if inside {
            for (k, &w) in weights.iter().enumerate() {
                let off = (k as i64 - margin) * stride;
                sum = step(sum, a[(i + off) as usize], w);
            }
        }
        sum
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_weights_from_each_template() {
        let src = "#define FAST 1000\n__kernel\nvoid box3(int width,  __global float* a, __global float* ret )\n{\n    if (x) {\n       sum = sum + a[i+(-1)] * 0.25;\n       sum = sum + a[i+(0)] * 0.5;\n       sum = sum + a[i+(1)] * 0.25;\n    }\n}\n__kernel void box3_fast(int width)\n{\n           sum = sum + a[j] * 0.25;\n}\n__kernel\nvoid g_h(int width)\n{\n    sum += inside ? a[i+(-1*height)] * 0.5 : 0.0f;\n    sum += inside ? a[i+(1*height)] * 0.5 : 0.0f;\n}\n__kernel\nvoid g(int vertical)\n{\n\t\tsum = sum + a[i+(-1*height)] * 0.5;\n\t\tsum = sum + a[i+(1*height)] * 0.5;\n    }\n    if (vertical) {\n\t\tsum = sum + a[i+(-1)] * 0.5;\n\t\tsum = sum + a[i+(1)] * 0.5;\n}\n";
        let k = parse(src).unwrap();
        assert_eq!(k[0], ("box3".to_string(), vec![0.25, 0.5, 0.25]));
        assert_eq!(k[1], ("box3_fast".to_string(), vec![]));
        assert_eq!(k[2], ("g_h".to_string(), vec![0.5, 0.5]));
        assert_eq!(k[3], ("g".to_string(), vec![0.5, 0.5]));
    }
}
