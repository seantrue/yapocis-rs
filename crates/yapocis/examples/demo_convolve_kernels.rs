//! `examples/demo_convolve_kernels.py`: 1-D convolution kernels from the
//! bundled `convolves` interface applied on the GPU: kernel properties, kernel
//! weights, smoothing and derivative filters, frequency response, and a
//! window-function comparison.
//!
//! The five matplotlib figures are redrawn with `examples/common/plot.rs`
//! (same content, not matplotlib's layout). `generate_kernel` is the Rust port
//! in `examples/common/kernels.rs`; NumPy's FFT in the window comparison is a
//! CPU DFT. NumPy's seeded noise (`np.random.seed(42)`) is replaced by the
//! examples' own seeded generator, so the noisy test signals differ in detail.
//!
//! cargo run -p yapocis --example demo_convolve_kernels --features metal --release

mod common;

use common::kernels::generate_kernel;
use common::plot::{with_alpha, Axes, Figure, Line, BLACK, BLUE, CYCLE, GREEN, GREY, ORANGE, RED};
use ndarray::ArrayD;
use yapocis::interfaces::convolves::Convolves;
use yapocis::rt::{CallArg, CallOptions, HostView};
use yapocis::{host, Device};

const EX: &str = "demo_convolve_kernels";

fn apply_kernel(prog: &Convolves, signal: &[f32], name: &str) -> yapocis::Result<Vec<f32>> {
    let shape = [signal.len()];
    let mut args = [CallArg::In(HostView::of(signal, &shape))];
    let out: ArrayD<f32> = host::one_output(prog.program().call(
        name,
        &mut args,
        &CallOptions::default(),
    )?)?;
    Ok(out.into_raw_vec_and_offset().0)
}

fn xs(n: usize) -> Vec<f64> {
    common::plot::linspace(0.0, 10.0, n)
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

fn range(v: &[f64]) -> (f64, f64) {
    let (lo, hi) = v
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    let pad = ((hi - lo) * 0.08).max(1e-3);
    (lo - pad, hi + pad)
}

fn create_test_signal(n: usize) -> Vec<f32> {
    let x = xs(n);
    let mut s = vec![0f32; n];
    for v in &mut s[40..80] {
        *v = 1.0;
    }
    for v in &mut s[120..160] {
        *v = 0.7;
    }
    for (i, v) in s[80..120].iter_mut().enumerate() {
        *v = 1.0 - i as f32 / 39.0;
    }
    let mut rng = common::Rng::new(42);
    for (i, v) in s.iter_mut().enumerate() {
        *v += 0.2 * (2.0 * std::f64::consts::PI * x[i]).sin() as f32 + 0.1 * rng.normal();
    }
    s
}

fn print_kernel_properties() {
    println!(
        "{}\nKERNEL MATHEMATICAL PROPERTIES\n{}\n",
        common::rule(70),
        common::rule(70)
    );
    println!(
        "{:<20} {:<6} {:<12} {:<12} {:<30}",
        "Kernel", "Size", "Sum", "Symmetric", "Purpose"
    );
    println!("{}", "-".repeat(80));
    for (t, size) in [
        ("box", 7),
        ("hamming", 9),
        ("binomial", 9),
        ("deriv1", 5),
        ("deriv2", 5),
        ("sharpen", 5),
        ("dog", 7),
    ] {
        let k = generate_kernel(t, size);
        let sum: f32 = k.iter().sum();
        let sym = k
            .iter()
            .zip(k.iter().rev())
            .all(|(a, b)| (a - b).abs() <= 1e-8 + 1e-5 * b.abs());
        let purpose = if (sum - 1.0).abs() < 1e-5 {
            "Smoothing (preserves DC)"
        } else if sum.abs() < 1e-5 {
            "Derivative/High-pass (removes DC)"
        } else {
            "General filtering"
        };
        let sym = if sym { "True" } else { "False" };
        println!("{t:<20} {size:<6} {sum:<12.6} {sym:<12} {purpose:<30}");
    }
    println!();
}

fn demo_kernel_visualization(device: &Device) -> yapocis::Result<()> {
    println!(
        "{}\nKERNEL WEIGHTS VISUALIZATION\n{}",
        common::rule(70),
        common::rule(70)
    );
    let kernels = [
        ("Box (7)", generate_kernel("box", 7)),
        ("Binomial (9)", generate_kernel("binomial", 9)),
        ("Hamming (11)", generate_kernel("hamming", 11)),
        ("First Derivative (5)", generate_kernel("deriv1", 5)),
        ("Second Derivative (5)", generate_kernel("deriv2", 5)),
        ("Sharpen (5)", generate_kernel("sharpen", 5)),
    ];
    let mut fig = Figure::new(device, 1500, 800)?;
    for (idx, (name, k)) in kernels.iter().enumerate() {
        let half = (k.len() / 2) as f64;
        let w = f64s(k);
        let (lo, hi) = range(&[w.clone(), vec![0.0]].concat());
        let ax = Axes::subplot(
            &fig,
            2,
            3,
            idx,
            (-half - 1.0, half + 1.0),
            (lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)),
        );
        for (i, &wi) in w.iter().enumerate() {
            let x = i as f64 - half;
            let face = if wi < 0.0 {
                with_alpha(RED, 0.7)
            } else {
                with_alpha(BLUE, 0.7)
            };
            ax.bar(&mut fig, x, 0.8, wi, face, Some(BLACK))?;
            let (px, py) = (ax.tx(x), ax.ty(wi));
            fig.text(
                px,
                if wi >= 0.0 { py + 2.0 } else { py - 2.0 },
                &format!("{wi:.3}"),
                8.0,
                BLACK,
                0.5,
                if wi >= 0.0 { 0.0 } else { 1.0 },
            )?;
        }
        ax.hline(&mut fig, 0.0, &Line::solid(BLACK, 1.0))?;
        let sum: f32 = k.iter().sum();
        ax.decorate(
            &mut fig,
            &format!("{name}  Sum: {sum:.3}"),
            "Position",
            "Weight",
        )?;
    }
    let path = common::out_dir(EX).join("kernel_weights.png");
    fig.save(&path)?;
    println!("Saved: {}\n", path.display());
    Ok(())
}

fn demo_smoothing_filters(device: &Device, prog: &Convolves, backend: &str) -> yapocis::Result<()> {
    println!(
        "{}\nSMOOTHING FILTERS DEMONSTRATION (using yapocis backend)\n{}",
        common::rule(70),
        common::rule(70)
    );
    let signal = create_test_signal(200);
    let x = xs(200);
    let kernels = [
        ("Original", None),
        ("Box (5)", Some("box5")),
        ("Box (11)", Some("box11")),
        ("Hamming (9)", Some("hamming9")),
        ("Hann (9)", Some("hann9")),
        ("Binomial (9)", Some("binomial9")),
    ];
    let mut fig = Figure::new(device, 1400, 1000)?;
    let yl = range(&f64s(&signal));
    for (idx, (name, kn)) in kernels.iter().enumerate() {
        let mut ax = Axes::subplot(&fig, 3, 2, idx, (0.0, 10.0), yl);
        match kn {
            None => {
                ax.line(
                    &mut fig,
                    &x,
                    &f64s(&signal),
                    &Line::solid(with_alpha(BLUE, 0.7), 1.0),
                    None,
                )?;
                ax.decorate(
                    &mut fig,
                    &format!("{name} (noisy input signal)"),
                    "Position",
                    "Amplitude",
                )?;
            }
            Some(k) => {
                let r = apply_kernel(prog, &signal, k)?;
                ax.line(
                    &mut fig,
                    &x,
                    &f64s(&signal),
                    &Line::solid(with_alpha(GREY, 0.3), 1.0),
                    Some("Original"),
                )?;
                ax.line(
                    &mut fig,
                    &x,
                    &f64s(&r),
                    &Line::solid(BLUE, 2.0),
                    Some("Filtered"),
                )?;
                ax.decorate(
                    &mut fig,
                    &format!("{name}  Backend: {backend}"),
                    "Position",
                    "Amplitude",
                )?;
                ax.legend(&mut fig, false)?;
            }
        }
    }
    let path = common::out_dir(EX).join("smoothing_filters.png");
    fig.save(&path)?;
    println!("Saved: {}\n", path.display());
    Ok(())
}

fn demo_derivative_filters(
    device: &Device,
    prog: &Convolves,
    backend: &str,
) -> yapocis::Result<()> {
    println!(
        "{}\nDERIVATIVE FILTERS DEMONSTRATION (using yapocis backend)\n{}",
        common::rule(70),
        common::rule(70)
    );
    let x = xs(200);
    let mut signal = vec![0f32; 200];
    for v in &mut signal[40..80] {
        *v = 1.0;
    }
    for v in &mut signal[100..140] {
        *v = 0.8;
    }
    for v in &mut signal[160..180] {
        *v = 0.5;
    }
    let mut rng = common::Rng::new(42);
    for v in &mut signal {
        *v += 0.02 * rng.normal();
    }
    let kernels = [
        ("Original", None),
        ("First Derivative (3-pt)", Some("deriv13")),
        ("First Derivative (5-pt)", Some("deriv15")),
        ("Sobel", Some("sobel3")),
        ("Second Derivative (3-pt)", Some("deriv23")),
        ("Second Derivative (5-pt)", Some("deriv25")),
    ];
    let mut fig = Figure::new(device, 1400, 1000)?;
    let circle: Vec<[f64; 2]> = (0..12)
        .map(|i| {
            let a = i as f64 / 12.0 * std::f64::consts::TAU;
            [a.cos(), a.sin()]
        })
        .collect();
    for (idx, (name, kn)) in kernels.iter().enumerate() {
        match kn {
            None => {
                let ax = Axes::subplot(&fig, 3, 2, idx, (0.0, 10.0), (-0.2, 1.2));
                let mut ax = ax;
                ax.line(&mut fig, &x, &f64s(&signal), &Line::solid(BLUE, 2.0), None)?;
                ax.decorate(
                    &mut fig,
                    &format!("{name} (step edges)"),
                    "Position",
                    "Amplitude",
                )?;
            }
            Some(k) => {
                let r = apply_kernel(prog, &signal, k)?;
                let rv = f64s(&r);
                let mut ax = Axes::subplot(&fig, 3, 2, idx, (0.0, 10.0), range(&rv));
                ax.line(
                    &mut fig,
                    &x,
                    &rv,
                    &Line::solid(RED, 2.0),
                    Some("Derivative"),
                )?;
                ax.hline(&mut fig, 0.0, &Line::dashed(with_alpha(BLACK, 0.5), 0.8))?;
                if name.contains("First") || name.contains("Sobel") {
                    let m = rv.iter().cloned().fold(0.0, |a: f64, v| a.max(v.abs()));
                    for (i, &v) in rv.iter().enumerate() {
                        if v.abs() > 0.3 * m {
                            ax.marker(&mut fig, x[i], v, &circle, 3.0, with_alpha(GREEN, 0.5))?;
                        }
                    }
                }
                ax.decorate(
                    &mut fig,
                    &format!("{name}  Backend: {backend}"),
                    "Position",
                    "Amplitude",
                )?;
                ax.legend(&mut fig, false)?;
            }
        }
    }
    let path = common::out_dir(EX).join("derivative_filters.png");
    fig.save(&path)?;
    println!("Saved: {}\n", path.display());
    Ok(())
}

fn demo_frequency_response(
    device: &Device,
    prog: &Convolves,
    backend: &str,
) -> yapocis::Result<()> {
    println!(
        "{}\nFREQUENCY RESPONSE DEMONSTRATION (using yapocis backend)\n{}",
        common::rule(70),
        common::rule(70)
    );
    let x = xs(500);
    let tp = 2.0 * std::f64::consts::PI;
    let low: Vec<f64> = x.iter().map(|&x| 0.3 * (tp * x).sin()).collect();
    let med: Vec<f64> = x.iter().map(|&x| 0.3 * (tp * 5.0 * x).sin()).collect();
    let high: Vec<f64> = x.iter().map(|&x| 0.3 * (tp * 15.0 * x).sin()).collect();
    let signal: Vec<f32> = (0..500)
        .map(|i| (low[i] + med[i] + high[i]) as f32)
        .collect();
    let mut fig = Figure::new(device, 1600, 800)?;
    let x100 = &x[..100];
    let mut ax = Axes::subplot(&fig, 2, 3, 0, (x100[0], x100[99]), (-1.0, 1.0));
    ax.line(
        &mut fig,
        x100,
        &f64s(&signal[..100]),
        &Line::solid(BLUE, 1.5),
        Some("Composite"),
    )?;
    ax.line(
        &mut fig,
        x100,
        &low[..100],
        &Line::dashed(with_alpha(GREEN, 0.5), 1.0),
        Some("Low freq"),
    )?;
    ax.line(
        &mut fig,
        x100,
        &med[..100],
        &Line::dashed(with_alpha(ORANGE, 0.5), 1.0),
        Some("Med freq"),
    )?;
    ax.line(
        &mut fig,
        x100,
        &high[..100],
        &Line::dashed(with_alpha(RED, 0.5), 1.0),
        Some("High freq"),
    )?;
    ax.decorate(
        &mut fig,
        "Original Signal (sum of 3 frequencies)",
        "Position",
        "Amplitude",
    )?;
    ax.legend(&mut fig, false)?;
    let filters = [
        ("Low-pass (Box 11)", "box11"),
        ("Low-pass (Hamming 13)", "hamming13"),
        ("High-pass (Sharpen 5)", "sharpen5"),
        ("Band-pass (DoG 9)", "dog9"),
    ];
    for (i, (name, k)) in filters.iter().enumerate() {
        let r = apply_kernel(prog, &signal, k)?;
        let both = [f64s(&signal[..100]), f64s(&r[..100])].concat();
        let mut ax = Axes::subplot(&fig, 2, 3, i + 1, (x100[0], x100[99]), range(&both));
        ax.line(
            &mut fig,
            x100,
            &f64s(&signal[..100]),
            &Line::solid(with_alpha(GREY, 0.3), 1.0),
            Some("Original"),
        )?;
        ax.line(
            &mut fig,
            x100,
            &f64s(&r[..100]),
            &Line::solid(BLUE, 2.0),
            Some("Filtered"),
        )?;
        ax.decorate(
            &mut fig,
            &format!("{name}  Backend: {backend}"),
            "Position",
            "Amplitude",
        )?;
        ax.legend(&mut fig, false)?;
    }
    let path = common::out_dir(EX).join("frequency_response.png");
    fig.save(&path)?;
    println!("Saved: {}\n", path.display());
    Ok(())
}

fn demo_window_comparison(device: &Device) -> yapocis::Result<()> {
    println!(
        "{}\nWINDOWING FUNCTIONS COMPARISON\n{}",
        common::rule(70),
        common::rule(70)
    );
    let size = 51;
    let windows = [
        ("Bartlett (Triangular)", generate_kernel("bartlett", size)),
        ("Hamming", generate_kernel("hamming", size)),
        ("Hann", generate_kernel("hann", size)),
        ("Blackman", generate_kernel("blackman", size)),
    ];
    let mut fig = Figure::new(device, 1400, 500)?;
    let x: Vec<f64> = (0..size).map(|i| i as f64).collect();
    let all: Vec<f64> = windows.iter().flat_map(|(_, w)| f64s(w)).collect();
    let mut ax1 = Axes::subplot(
        &fig,
        1,
        2,
        0,
        (0.0, (size - 1) as f64),
        (0.0, range(&all).1),
    );
    let dot: Vec<[f64; 2]> = (0..8)
        .map(|i| {
            let a = i as f64 / 8.0 * std::f64::consts::TAU;
            [a.cos(), a.sin()]
        })
        .collect();
    for (i, (name, w)) in windows.iter().enumerate() {
        let c = with_alpha(CYCLE[i], 0.7);
        ax1.line(&mut fig, &x, &f64s(w), &Line::solid(c, 2.0), Some(name))?;
        for (xi, &wi) in x.iter().zip(w) {
            ax1.marker(&mut fig, *xi, wi as f64, &dot, 3.0, c)?;
        }
    }
    ax1.decorate(
        &mut fig,
        "Window Shapes (Time Domain)",
        "Sample Index",
        "Amplitude",
    )?;
    ax1.legend(&mut fig, false)?;
    let mut ax2 = Axes::subplot(&fig, 1, 2, 1, (0.0, 0.5), (-100.0, 5.0));
    for (i, (name, w)) in windows.iter().enumerate() {
        // 512-point zero-padded DFT, first 256 bins, normalised to 0 dB
        let n = 512;
        let mag: Vec<f64> = (0..256)
            .map(|k| {
                let (mut re, mut im) = (0.0, 0.0);
                for (j, &v) in w.iter().enumerate() {
                    let a = -std::f64::consts::TAU * (k * j) as f64 / n as f64;
                    re += v as f64 * a.cos();
                    im += v as f64 * a.sin();
                }
                20.0 * ((re * re + im * im).sqrt() + 1e-10).log10()
            })
            .collect();
        let peak = mag.iter().cloned().fold(f64::MIN, f64::max);
        let freq: Vec<f64> = (0..256).map(|k| k as f64 / n as f64).collect();
        let db: Vec<f64> = mag.iter().map(|m| (m - peak).max(-100.0)).collect();
        ax2.line(
            &mut fig,
            &freq,
            &db,
            &Line::solid(with_alpha(CYCLE[i], 0.7), 2.0),
            Some(name),
        )?;
    }
    ax2.decorate(
        &mut fig,
        "Frequency Response (Magnitude)",
        "Normalized Frequency",
        "Magnitude (dB)",
    )?;
    ax2.legend(&mut fig, false)?;
    let path = common::out_dir(EX).join("window_comparison.png");
    fig.save(&path)?;
    println!("Saved: {}\n", path.display());
    Ok(())
}

fn main() -> yapocis::Result<()> {
    println!("Loading convolves interface...");
    let device = common::device();
    let prog = Convolves::load(&device)?;
    let backend = device.backend().name();
    println!("Backend: {backend}");
    println!("Device: {}\n", device.name());
    println!("\n{}\nCONVOLVES KERNEL DEMONSTRATION", common::rule(70));
    println!("Signal Processing with 1D Convolution Kernels");
    println!(
        "Using yapocis backend-accelerated convolution\n{}\n",
        common::rule(70)
    );
    print_kernel_properties();
    demo_kernel_visualization(&device)?;
    demo_smoothing_filters(&device, &prog, backend)?;
    demo_derivative_filters(&device, &prog, backend)?;
    demo_frequency_response(&device, &prog, backend)?;
    demo_window_comparison(&device)?;
    println!(
        "{}\nALL DEMONSTRATIONS COMPLETE\n{}\n",
        common::rule(70),
        common::rule(70)
    );
    println!("Backend used: {backend}");
    println!("Device: {}\n", device.name());
    println!(
        "Generated visualizations in {}:",
        common::out_dir(EX).display()
    );
    for f in [
        "kernel_weights",
        "smoothing_filters",
        "derivative_filters",
        "frequency_response",
        "window_comparison",
    ] {
        println!("  - {f}.png");
    }
    Ok(())
}
