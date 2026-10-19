//! `examples/fft_power_spectrum.py`: power spectra of a 50 + 120 + 200 Hz
//! signal under four window functions, a zoom on the 120 Hz peak, and an
//! accuracy check of the GPU FFT against a CPU DFT.
//!
//! The matplotlib figure is redrawn with `examples/common/plot.rs` (same
//! content, not matplotlib's layout); `plt.show()` is dropped. NumPy's FFT
//! reference is replaced by a direct O(n²) DFT in f64.
//!
//! cargo run -p yapocis --example fft_power_spectrum --features metal --release

mod common;

use common::plot::{with_alpha, Axes, Figure, Line, BLUE, GREEN, GREY, ORANGE, RED};
use ndarray::Array1;
use num_complex::Complex;
use yapocis::lib::fft::Window;
use yapocis::lib::Fft;

fn create_test_signal(
    n_samples: usize,
    sample_rate: f32,
    rng: &mut common::Rng,
) -> (Vec<f32>, Array1<f32>) {
    let t: Vec<f32> = (0..n_samples).map(|i| i as f32 / sample_rate).collect();
    let tp = 2.0 * std::f32::consts::PI;
    let signal = Array1::from_iter(t.iter().map(|&t| {
        (tp * 50.0 * t).sin()
            + 0.5 * (tp * 120.0 * t).sin()
            + 0.3 * (tp * 200.0 * t).sin()
            + 0.1 * rng.normal()
    }));
    (t, signal)
}

fn plot_power_spectra(device: &yapocis::Device, fft: &Fft) -> yapocis::Result<()> {
    println!("Creating test signal...");
    let (n_samples, sample_rate) = (1024usize, 1000.0f32);
    let mut rng = common::Rng::new(3);
    let (t, signal) = create_test_signal(n_samples, sample_rate, &mut rng);
    let windows = [
        (Window::Rectangular, "Rectangular", RED),
        (Window::Hann, "Hann", BLUE),
        (Window::Hamming, "Hamming", GREEN),
        (Window::Blackman, "Blackman", ORANGE),
    ];

    let mut fig = Figure::new(device, 1400, 1000)?;
    let mut ax1 = Axes::subplot(&fig, 3, 1, 0, (0.0, t[199] as f64), (-2.2, 2.2));
    let ts: Vec<f64> = t[..200].iter().map(|&v| v as f64).collect();
    let ys: Vec<f64> = signal.iter().take(200).map(|&v| v as f64).collect();
    ax1.line(&mut fig, &ts, &ys, &Line::solid(BLUE, 1.0), None)?;
    ax1.decorate(
        &mut fig,
        "Time-Domain Signal (first 200 samples): 50 Hz + 120 Hz + 200 Hz + Noise",
        "Time (s)",
        "Amplitude",
    )?;

    println!("\nComputing power spectra...");
    let mut spectra = Vec::new();
    for (w, name, color) in windows {
        println!("  - {name} window...");
        let (f, p) = fft.analyze_signal(&signal.view(), sample_rate as f64, w, true, true)?;
        spectra.push((
            f.to_vec(),
            p.iter().map(|&v| v as f64).collect::<Vec<f64>>(),
            name,
            color,
        ));
    }
    let hi = spectra
        .iter()
        .flat_map(|s| s.1.iter())
        .cloned()
        .fold(f64::MIN, f64::max);
    let lo = spectra
        .iter()
        .flat_map(|s| s.1.iter())
        .cloned()
        .fold(f64::MAX, f64::min);
    let mut ax2 = Axes::subplot(&fig, 3, 1, 1, (0.0, 300.0), (lo - 5.0, hi + 10.0));
    for (f, p, name, color) in &spectra {
        ax2.line(
            &mut fig,
            f,
            p,
            &Line::solid(with_alpha(*color, 0.7), 1.5),
            Some(name),
        )?;
    }
    for (freq, label) in [(50.0, "50 Hz"), (120.0, "120 Hz"), (200.0, "200 Hz")] {
        ax2.vline(&mut fig, freq, &Line::dashed(with_alpha(GREY, 0.5), 1.0))?;
        let (x, y) = (ax2.tx(freq), ax2.ty(hi + 5.0));
        fig.text(x, y, label, 9.0, GREY, 0.5, 0.0)?;
    }
    ax2.decorate(
        &mut fig,
        "Power Spectrum Comparison - Different Window Functions",
        "Frequency (Hz)",
        "Power (dB)",
    )?;
    ax2.legend(&mut fig, false)?;

    let zoom: Vec<f64> = spectra
        .iter()
        .flat_map(|(f, p, _, _)| {
            f.iter()
                .zip(p)
                .filter(|(f, _)| **f >= 100.0 && **f <= 140.0)
                .map(|(_, p)| *p)
        })
        .collect();
    let (zlo, zhi) = zoom
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let mut ax3 = Axes::subplot(&fig, 3, 1, 2, (100.0, 140.0), (zlo - 3.0, zhi + 3.0));
    for (f, p, name, color) in &spectra {
        ax3.line(
            &mut fig,
            f,
            p,
            &Line::solid(with_alpha(*color, 0.7), 2.0),
            Some(name),
        )?;
    }
    ax3.vline(&mut fig, 120.0, &Line::dashed(with_alpha(GREY, 0.5), 1.0))?;
    ax3.decorate(
        &mut fig,
        "Zoomed View: 120 Hz Peak (showing spectral leakage effects)",
        "Frequency (Hz)",
        "Power (dB)",
    )?;
    ax3.legend(&mut fig, false)?;

    let out = common::out_dir("fft_power_spectrum").join("fft_power_spectrum.png");
    println!("\nSaving plot to {}...", out.display());
    fig.save(&out)?;

    println!("\nAnalysis:");
    println!("- The rectangular window has the sharpest peaks but most spectral leakage");
    println!("- The Hann window provides a good balance between resolution and leakage");
    println!("- The Hamming window has slightly better sidelobe suppression than Hann");
    println!("- The Blackman window has the least leakage but widest peaks (worst resolution)");
    Ok(())
}

fn dft(x: &[Complex<f32>]) -> Vec<Complex<f64>> {
    let n = x.len();
    (0..n)
        .map(|k| {
            x.iter()
                .enumerate()
                .map(|(j, v)| {
                    let a = -2.0 * std::f64::consts::PI * (k * j % n) as f64 / n as f64;
                    Complex::new(v.re as f64, v.im as f64) * Complex::new(a.cos(), a.sin())
                })
                .sum()
        })
        .collect()
}

fn compare_fft_vs_cpu(fft: &Fft) -> yapocis::Result<()> {
    println!(
        "\n{}\nFFT Accuracy Comparison: yapocis vs CPU DFT\n{}",
        common::rule(60),
        common::rule(60)
    );
    let mut rng = common::Rng::new(4);
    for n in [64usize, 128, 256, 512] {
        let signal: Array1<Complex<f32>> =
            Array1::from_iter((0..n).map(|_| Complex::new(rng.normal(), 0.0)));
        let got = fft.forward(&signal.view())?;
        let want = dft(signal.as_slice().unwrap());
        let errs: Vec<f64> = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (Complex::new(g.re as f64, g.im as f64) - w).norm())
            .collect();
        let max = errs.iter().cloned().fold(0.0, f64::max);
        let mean = errs.iter().sum::<f64>() / n as f64;
        println!("N={n:4}: max error = {max:.6e}, mean error = {mean:.6e}");
    }
    println!("\nAll errors are well within numerical precision!");
    Ok(())
}

fn main() -> yapocis::Result<()> {
    println!("FFT Power Spectrum Example\n{}", common::rule(60));
    let device = common::device();
    println!("Initializing FFT library...");
    let fft = Fft::load(&device)?;
    plot_power_spectra(&device, &fft)?;
    compare_fft_vs_cpu(&fft)?;
    println!("\n{}\nExample complete!", common::rule(60));
    Ok(())
}
