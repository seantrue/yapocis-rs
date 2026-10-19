//! `examples/median_performance.py`: 3×3 median filter timing over image sizes
//! from 100² to 1000², async (resident ping-pong, one wait) against sync (a
//! wait after every pass), per backend, against a CPU baseline.
//!
//! Differences from the Python example: the CPU baseline is a straightforward
//! Rust 3×3 median with scipy's `reflect` boundary instead of
//! `scipy.ndimage.median_filter`; each compiled-in backend is tested on one
//! device (its default, or the first `--devices` name that matches) because the
//! Rust runtime has no device-listing API; and the seaborn figure is redrawn
//! with `examples/common/plot.rs` (same three panels, not seaborn's styling).
//!
//! cargo run -p yapocis --example median_performance --features metal,ocl --release -- \
//!     [--backends metal,ocl] [--devices NAME,...] [--iterations 20] [--samples 20]

mod common;

use common::plot::{with_alpha, Axes, Figure, Line, CYCLE, GREY};
use ndarray::{Array2, ArrayD, IxDyn};
use yapocis::lib::Median;
use yapocis::rt::{CallArg, CallOptions};
use yapocis::{available_backends, Backend, Device};

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn reflect(i: isize, n: usize) -> usize {
    // scipy 'reflect': (d c b a | a b c d | d c b a)
    if i < 0 {
        (-i - 1) as usize
    } else if i as usize >= n {
        2 * n - 1 - i as usize
    } else {
        i as usize
    }
}

fn median3x3_cpu(img: &Array2<f32>, iterations: usize) -> Array2<f32> {
    let (h, w) = img.dim();
    let mut cur = img.clone();
    for _ in 0..iterations {
        let mut next = Array2::<f32>::zeros((h, w));
        for r in 0..h {
            for c in 0..w {
                let mut v = [0f32; 9];
                let mut k = 0;
                for dr in -1..=1isize {
                    for dc in -1..=1isize {
                        v[k] = cur[[reflect(r as isize + dr, h), reflect(c as isize + dc, w)]];
                        k += 1;
                    }
                }
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                next[[r, c]] = v[4];
            }
        }
        cur = next;
    }
    cur
}

/// Sync mode: same resident ping-pong, but every pass waits for completion.
fn median_sync(
    median: &Median,
    img: &ArrayD<f32>,
    iterations: usize,
) -> yapocis::Result<ArrayD<f32>> {
    let device = median.kernels.device().clone();
    let mut input = device.upload(img.as_slice().unwrap(), img.shape())?;
    let mut output = device.alloc::<f32>(img.shape())?;
    for _ in 0..iterations {
        let mut args = [
            CallArg::Buffer(input.as_any()),
            CallArg::Buffer(output.as_any()),
        ];
        median
            .kernels
            .program()
            .call("median3x3_res", &mut args, &CallOptions::default())?;
        std::mem::swap(&mut input, &mut output);
    }
    Ok(ArrayD::from_shape_vec(IxDyn(img.shape()), input.to_vec()?)?)
}

fn time<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = std::time::Instant::now();
    let v = f();
    (v, t.elapsed().as_secs_f64())
}

struct Run {
    label: String,
    areas: Vec<f64>,
    asyn: Vec<f64>,
    sync: Vec<f64>,
}

fn main() -> yapocis::Result<()> {
    let iterations: usize = arg("--iterations")
        .map(|s| s.parse().unwrap())
        .unwrap_or(20);
    let samples: usize = arg("--samples").map(|s| s.parse().unwrap()).unwrap_or(20);
    let backend_filter: Vec<String> = arg("--backends")
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
        .unwrap_or_default();
    let device_filter: Vec<String> = arg("--devices")
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
        .unwrap_or_default();
    let (smallest, largest) = (100.0f64, 1000.0f64);

    println!(
        "{}\nMulti-Backend Median Filter Performance Comparison\n{}",
        common::rule(70),
        common::rule(70)
    );
    println!("Image sizes: {smallest}x{smallest} to {largest}x{largest} ({samples} samples)");
    println!("Iterations per test: {iterations}");
    println!("Testing: Async vs Sync resident buffers (ping-pong)\n");

    let widths: Vec<usize> = (0..samples)
        .map(|i| {
            let e = smallest.log2()
                + (largest.log2() - smallest.log2()) * i as f64 / (samples - 1).max(1) as f64;
            2f64.powf(e) as usize
        })
        .collect();
    let mut rng = common::Rng::new(11);
    let images: Vec<Array2<f32>> = widths
        .iter()
        .map(|&w| Array2::from_shape_fn((w, w), |_| rng.uniform()))
        .collect();

    println!("\n{}\nCPU Baseline\n{}", common::rule(70), common::rule(70));
    println!("\nBenchmarking CPU 3x3 median (sync)...");
    let mut cpu_areas = Vec::new();
    let mut cpu_times = Vec::new();
    for (w, img) in widths.iter().zip(&images) {
        median3x3_cpu(img, 1);
        let (_, t) = time(|| median3x3_cpu(img, iterations));
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

    let mut results = Vec::new();
    for backend in backends {
        let selector = device_filter.first().map(|s| s.as_str());
        let device = match Device::open(backend, selector) {
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
        let median = Median::load(&device)?;
        let mut r = Run {
            label: label.clone(),
            areas: vec![],
            asyn: vec![],
            sync: vec![],
        };
        for (mode, is_async) in [("async", true), ("sync", false)] {
            println!("\nBenchmarking {label} ({mode})...");
            for (w, img) in widths.iter().zip(&images) {
                let a = img.clone().into_dyn();
                let run = |n| {
                    if is_async {
                        median.median3x3(&a.view(), n)
                    } else {
                        median_sync(&median, &a, n)
                    }
                };
                run(1)?;
                let (out, t) = time(|| run(iterations));
                out?;
                if is_async {
                    r.areas.push((w * w) as f64 / 1000.0);
                    r.asyn.push(t);
                } else {
                    r.sync.push(t);
                }
                println!("  {w:4}x{w:4}: {t:.4}s");
            }
        }
        results.push(r);
    }
    if results.is_empty() {
        println!("\nNo successful benchmarks, cannot plot.");
        std::process::exit(1);
    }

    println!(
        "\n{}\nGenerating plots...\n{}",
        common::rule(70),
        common::rule(70)
    );
    let device = common::device();
    let mut fig = Figure::new(&device, 1400, 1400)?;
    let xr = (0.0, cpu_areas.last().copied().unwrap_or(1000.0) * 1.02);
    let all_t: Vec<f64> = cpu_times
        .iter()
        .chain(results.iter().flat_map(|r| r.asyn.iter().chain(&r.sync)))
        .cloned()
        .collect();
    let (tlo, thi) = all_t
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let mut ax1 = Axes::subplot(&fig, 3, 1, 0, xr, (tlo * 0.7, thi * 1.5));
    ax1.log_y = true;
    ax1.line(
        &mut fig,
        &cpu_areas,
        &cpu_times,
        &Line::dashed(with_alpha(GREY, 0.7), 2.5),
        Some("CPU (baseline)"),
    )?;
    for (i, r) in results.iter().enumerate() {
        let c = CYCLE[i % CYCLE.len()];
        ax1.line(
            &mut fig,
            &r.areas,
            &r.asyn,
            &Line::solid(c, 2.0),
            Some(&format!("{} (async)", r.label)),
        )?;
        ax1.line(
            &mut fig,
            &r.areas,
            &r.sync,
            &Line::dashed(with_alpha(c, 0.6), 2.0),
            Some(&format!("{} (sync)", r.label)),
        )?;
    }
    ax1.decorate(&mut fig, &format!("3x3 Median Filter Performance ({iterations} iterations): Async vs Sync Resident Buffers"), "", "Execution Time (seconds)")?;
    ax1.legend(&mut fig, true)?;

    let speed: Vec<Vec<f64>> = results
        .iter()
        .map(|r| r.asyn.iter().zip(&cpu_times).map(|(g, c)| c / g).collect())
        .collect();
    let all: Vec<f64> = speed.iter().flatten().cloned().chain([1.0]).collect();
    let (slo, shi) = all
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let mut ax2 = Axes::subplot(&fig, 3, 1, 1, xr, (slo * 0.7, shi * 1.5));
    ax2.log_y = true;
    for (i, (r, s)) in results.iter().zip(&speed).enumerate() {
        ax2.line(
            &mut fig,
            &r.areas,
            s,
            &Line::solid(CYCLE[i % CYCLE.len()], 2.0),
            Some(&format!("{} (async)", r.label)),
        )?;
    }
    ax2.hline(&mut fig, 1.0, &Line::dashed(with_alpha(GREY, 0.7), 1.5))?;
    ax2.add_legend("CPU baseline", GREY);
    ax2.decorate(
        &mut fig,
        "GPU Acceleration (async mode, higher is better)",
        "",
        "Speedup vs CPU",
    )?;
    ax2.legend(&mut fig, true)?;

    let gain: Vec<Vec<f64>> = results
        .iter()
        .map(|r| r.sync.iter().zip(&r.asyn).map(|(s, a)| s / a).collect())
        .collect();
    let all: Vec<f64> = gain.iter().flatten().cloned().chain([1.0]).collect();
    let (glo, ghi) = all
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let mut ax3 = Axes::subplot(
        &fig,
        3,
        1,
        2,
        xr,
        (
            glo - 0.1 * (ghi - glo).max(0.1),
            ghi + 0.1 * (ghi - glo).max(0.1),
        ),
    );
    for (i, (r, g)) in results.iter().zip(&gain).enumerate() {
        ax3.line(
            &mut fig,
            &r.areas,
            g,
            &Line::solid(CYCLE[i % CYCLE.len()], 2.0),
            Some(&r.label),
        )?;
    }
    ax3.hline(&mut fig, 1.0, &Line::dashed(with_alpha(GREY, 0.7), 1.5))?;
    ax3.add_legend("No benefit", GREY);
    ax3.decorate(
        &mut fig,
        "Async vs Sync Performance Gain (higher is better)",
        "Image Size (kilopixels)",
        "Async Speedup",
    )?;
    ax3.legend(&mut fig, true)?;
    let out = common::out_dir("median_performance").join("median_performance.png");
    fig.save(&out)?;

    println!(
        "\n{}\nPerformance Summary (at largest image size)\n{}",
        common::rule(70),
        common::rule(70)
    );
    println!(
        "{:<35} {:>11} {:>11} {:>10} {:>12}",
        "Backend:Device", "Async (s)", "Sync (s)", "vs CPU", "Async Gain"
    );
    println!("{}", "-".repeat(70));
    println!(
        "{:<35} {:>11} {:>11.4} {:>10} {:>12}",
        "CPU baseline",
        "-",
        cpu_times.last().unwrap(),
        "1.00x",
        "-"
    );
    for (r, (s, g)) in results.iter().zip(speed.iter().zip(&gain)) {
        println!(
            "{:<35} {:>11.4} {:>11.4} {:>9.2}x {:>11.2}x",
            r.label,
            r.asyn.last().unwrap(),
            r.sync.last().unwrap(),
            s.last().unwrap(),
            g.last().unwrap()
        );
    }
    println!("{}", common::rule(70));
    Ok(())
}
