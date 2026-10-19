//! `examples/zcs_performance.py`: difference-of-Gaussians zero crossings
//! (`Gaussian::zcsdog`) timed over image sizes from 100² to 1000², per backend,
//! against a CPU baseline, on a synthetic disc-and-cross image.
//!
//! Differences from the Python example: the CPU baseline is a Rust port of the
//! scipy reference (`gaussian_filter` with `truncate=4`, `mode='reflect'`,
//! sign-change zero crossings, `np.gradient`); the Rust `zcsdog` always runs
//! its device passes asynchronously with a single wait, so there is no separate
//! sync/async pair and the "async gain" panel is dropped; one device per
//! backend (default, or the first `--devices` match); the seaborn figure is
//! redrawn with `examples/common/plot.rs`. The last zero-crossing map is also
//! written as a PNG.
//!
//! cargo run -p yapocis --example zcs_performance --features metal,ocl --release -- \
//!     [--backends metal,ocl] [--devices NAME,...] [--iterations 10] [--samples 20] [--scale 2]

mod common;

use common::plot::{with_alpha, Axes, Figure, Line, CYCLE, GREY};
use ndarray::Array2;
use yapocis::lib::Gaussian;
use yapocis::{available_backends, Backend, Device};

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn reflect(i: isize, n: usize) -> usize {
    let n = n as isize;
    let mut i = i;
    while i < 0 || i >= n {
        i = if i < 0 { -i - 1 } else { 2 * n - 1 - i };
    }
    i as usize
}

/// `scipy.ndimage.gaussian_filter(img, sigma)` (truncate 4, reflect), separable.
fn gaussian_filter(img: &Array2<f32>, sigma: f64) -> Array2<f32> {
    let radius = (4.0 * sigma + 0.5) as isize;
    let mut k: Vec<f64> = (-radius..=radius)
        .map(|x| (-(x * x) as f64 / (2.0 * sigma * sigma)).exp())
        .collect();
    let s: f64 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    let (h, w) = img.dim();
    let mut tmp = Array2::<f32>::zeros((h, w));
    for r in 0..h {
        for c in 0..w {
            let mut acc = 0.0;
            for (j, kv) in k.iter().enumerate() {
                acc += *kv * img[[reflect(r as isize + j as isize - radius, h), c]] as f64;
            }
            tmp[[r, c]] = acc as f32;
        }
    }
    let mut out = Array2::<f32>::zeros((h, w));
    for r in 0..h {
        for c in 0..w {
            let mut acc = 0.0;
            for (j, kv) in k.iter().enumerate() {
                acc += *kv * tmp[[r, reflect(c as isize + j as isize - radius, w)]] as f64;
            }
            out[[r, c]] = acc as f32;
        }
    }
    out
}

/// `np.gradient` along `axis` (central differences, one-sided at the edges).
fn gradient(a: &Array2<f32>, axis: usize) -> Array2<f32> {
    let (h, w) = a.dim();
    Array2::from_shape_fn((h, w), |(r, c)| {
        let (i, n) = if axis == 0 { (r, h) } else { (c, w) };
        let at = |k: usize| if axis == 0 { a[[k, c]] } else { a[[r, k]] };
        if i == 0 {
            at(1) - at(0)
        } else if i == n - 1 {
            at(n - 1) - at(n - 2)
        } else {
            (at(i + 1) - at(i - 1)) / 2.0
        }
    })
}

fn zcsdog_cpu(
    img: &Array2<f32>,
    scale: i32,
    iterations: usize,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let (s1, s2) = (1.6f64.powi(scale), 1.6f64.powi(scale + 1));
    let mut res = None;
    for _ in 0..iterations {
        let smaller = gaussian_filter(img, s1);
        let larger = gaussian_filter(img, s2);
        let dog = &smaller - &larger;
        let (h, w) = dog.dim();
        let mut zc = Array2::<f32>::zeros((h, w));
        for r in 0..h {
            for c in 0..w {
                if r + 1 < h && dog[[r, c]] * dog[[r + 1, c]] < 0.0 {
                    zc[[r, c]] = 1.0;
                }
                if c + 1 < w && dog[[r, c]] * dog[[r, c + 1]] < 0.0 {
                    zc[[r, c]] = 1.0;
                }
            }
        }
        let (g0, g1) = (gradient(&smaller, 0), gradient(&smaller, 1));
        let grad = g0.mapv(f32::abs) + g1.mapv(f32::abs);
        let theta = ndarray::Zip::from(&g0)
            .and(&g1)
            .map_collect(|a, b| a.atan2(*b));
        res = Some((zc, grad, theta));
    }
    res.unwrap()
}

fn test_image(w: usize) -> Array2<f32> {
    let (center, radius) = (w / 2, w / 4);
    let mut img = Array2::from_shape_fn((w, w), |(y, x)| {
        let (dx, dy) = (x as isize - center as isize, y as isize - center as isize);
        if dx * dx + dy * dy <= (radius * radius) as isize {
            1.0
        } else {
            0.0
        }
    });
    for i in center - 5..center + 5 {
        for j in 0..w {
            img[[i, j]] = 0.5;
            img[[j, i]] = 0.5;
        }
    }
    img
}

fn time<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = std::time::Instant::now();
    let v = f();
    (v, t.elapsed().as_secs_f64())
}

fn main() -> yapocis::Result<()> {
    let iterations: usize = arg("--iterations")
        .map(|s| s.parse().unwrap())
        .unwrap_or(10);
    let samples: usize = arg("--samples").map(|s| s.parse().unwrap()).unwrap_or(20);
    let scale: i32 = arg("--scale").map(|s| s.parse().unwrap()).unwrap_or(2);
    let backend_filter: Vec<String> = arg("--backends")
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
        .unwrap_or_default();
    let device_filter: Vec<String> = arg("--devices")
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
        .unwrap_or_default();
    let (smallest, largest) = (100.0f64, 1000.0f64);

    println!(
        "{}\nMulti-Backend Multiscale Zero-Crossing Performance Comparison\n{}",
        common::rule(70),
        common::rule(70)
    );
    println!("Image sizes: {smallest}x{smallest} to {largest}x{largest} ({samples} samples)");
    println!("Iterations per test: {iterations}");
    println!(
        "DoG scale level: {scale} (sigma ratio: 1.6^{scale} vs 1.6^{})",
        scale + 1
    );
    println!("Testing: async resident buffers\n");

    let widths: Vec<usize> = (0..samples)
        .map(|i| {
            let e = smallest.log2()
                + (largest.log2() - smallest.log2()) * i as f64 / (samples - 1).max(1) as f64;
            2f64.powf(e) as usize
        })
        .collect();
    let images: Vec<Array2<f32>> = widths.iter().map(|&w| test_image(w)).collect();

    println!("\n{}\nCPU Baseline\n{}", common::rule(70), common::rule(70));
    println!("\nBenchmarking CPU (scale={scale})...");
    let (mut cpu_areas, mut cpu_times) = (Vec::new(), Vec::new());
    for (w, img) in widths.iter().zip(&images) {
        zcsdog_cpu(img, scale, 1);
        let (_, t) = time(|| zcsdog_cpu(img, scale, iterations));
        cpu_areas.push((w * w) as f64 / 1000.0);
        cpu_times.push(t);
        println!("  {w:4}x{w:4}: {t:.4}s");
    }

    let backends: Vec<Backend> = available_backends()
        .into_iter()
        .filter(|b| {
            backend_filter.is_empty()
                || backend_filter.iter().any(|f| Backend::parse(f) == Some(*b))
        })
        .collect();
    println!(
        "\n{}\nTesting backends: {}\n{}",
        common::rule(70),
        backends
            .iter()
            .map(|b| b.name())
            .collect::<Vec<_>>()
            .join(", "),
        common::rule(70)
    );
    let mut results: Vec<(String, Vec<f64>, Vec<f64>)> = Vec::new();
    let mut last_zc = None;
    for backend in backends {
        let device = match Device::open(backend, device_filter.first().map(|s| s.as_str())) {
            Ok(d) => d,
            Err(e) => {
                println!(
                    "\nNo devices found for {}, skipping... ({e})",
                    backend.name()
                );
                continue;
            }
        };
        let label = format!("{}: {}", backend.name().to_uppercase(), device.name());
        let g = Gaussian::load(&device)?;
        println!("\nBenchmarking {label} (scale={scale}, async)...");
        let (mut areas, mut times) = (Vec::new(), Vec::new());
        for (w, img) in widths.iter().zip(&images) {
            let a = img.clone().into_dyn();
            g.zcsdog(&a.view(), scale as usize, None, true, true)?;
            let (r, t) = time(|| -> yapocis::Result<_> {
                let mut out = None;
                for _ in 0..iterations {
                    out = Some(g.zcsdog(&a.view(), scale as usize, None, true, true)?);
                }
                Ok(out)
            });
            last_zc = r?.map(|(zc, _, _)| zc);
            areas.push((w * w) as f64 / 1000.0);
            times.push(t);
            println!("  {w:4}x{w:4}: {t:.4}s");
        }
        results.push((label, areas, times));
    }
    if results.is_empty() {
        println!("\nNo successful benchmarks, cannot plot.");
        std::process::exit(1);
    }
    if let Some(zc) = last_zc {
        common::show("zcs_performance", "zero_crossings_largest", &zc.into_dyn());
    }

    println!(
        "\n{}\nGenerating plots...\n{}",
        common::rule(70),
        common::rule(70)
    );
    let device = common::device();
    let mut fig = Figure::new(&device, 1400, 950)?;
    let xr = (0.0, cpu_areas.last().copied().unwrap_or(1000.0) * 1.02);
    let all: Vec<f64> = cpu_times
        .iter()
        .chain(results.iter().flat_map(|r| r.2.iter()))
        .cloned()
        .collect();
    let (lo, hi) = all
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let mut ax1 = Axes::subplot(&fig, 2, 1, 0, xr, (lo * 0.7, hi * 1.5));
    ax1.log_y = true;
    ax1.line(
        &mut fig,
        &cpu_areas,
        &cpu_times,
        &Line::dashed(with_alpha(GREY, 0.7), 2.5),
        Some("CPU (baseline)"),
    )?;
    for (i, (label, areas, times)) in results.iter().enumerate() {
        ax1.line(
            &mut fig,
            areas,
            times,
            &Line::solid(CYCLE[i % CYCLE.len()], 2.0),
            Some(&format!("{label} (async)")),
        )?;
    }
    ax1.decorate(
        &mut fig,
        &format!("Multiscale Zero-Crossing Performance (scale={scale}, {iterations} iterations)"),
        "",
        "Execution Time (seconds)",
    )?;
    ax1.legend(&mut fig, true)?;
    let speed: Vec<Vec<f64>> = results
        .iter()
        .map(|r| r.2.iter().zip(&cpu_times).map(|(g, c)| c / g).collect())
        .collect();
    let all: Vec<f64> = speed.iter().flatten().cloned().chain([1.0]).collect();
    let (slo, shi) = all
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let mut ax2 = Axes::subplot(&fig, 2, 1, 1, xr, (slo * 0.7, shi * 1.5));
    ax2.log_y = true;
    for (i, ((label, areas, _), s)) in results.iter().zip(&speed).enumerate() {
        ax2.line(
            &mut fig,
            areas,
            s,
            &Line::solid(CYCLE[i % CYCLE.len()], 2.0),
            Some(label),
        )?;
    }
    ax2.hline(&mut fig, 1.0, &Line::dashed(with_alpha(GREY, 0.7), 1.5))?;
    ax2.add_legend("CPU baseline", GREY);
    ax2.decorate(
        &mut fig,
        "GPU Acceleration (higher is better)",
        "Image Size (kilopixels)",
        "Speedup vs CPU",
    )?;
    ax2.legend(&mut fig, true)?;
    fig.save(&common::out_dir("zcs_performance").join("zcs_performance.png"))?;

    println!(
        "\n{}\nPerformance Summary (at largest image size)\n{}",
        common::rule(70),
        common::rule(70)
    );
    println!(
        "{:<35} {:>11} {:>10}",
        "Backend:Device", "Time (s)", "vs CPU"
    );
    println!("{}", "-".repeat(60));
    println!(
        "{:<35} {:>11.4} {:>10}",
        "CPU baseline",
        cpu_times.last().unwrap(),
        "1.00x"
    );
    for ((label, _, times), s) in results.iter().zip(&speed) {
        println!(
            "{:<35} {:>11.4} {:>9.2}x",
            label,
            times.last().unwrap(),
            s.last().unwrap()
        );
    }
    println!("{}", common::rule(70));
    Ok(())
}
