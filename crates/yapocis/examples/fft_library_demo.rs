//! `examples/fft_library_demo.py`: five demos of the high-level FFT library:
//! spectral analysis with peaks, a spectrogram, band-pass filtering, signal
//! metrics (energy, RMS, SNR), and coherence.
//!
//! The four matplotlib figures are redrawn with `examples/common/plot.rs`
//! (same content, not matplotlib's layout; no colour bar).
//!
//! cargo run -p yapocis --example fft_library_demo --features metal --release

mod common;

use common::plot::{with_alpha, Axes, Figure, Line, BLUE, GREEN, RED};
use ndarray::Array1;
use yapocis::lib::fft::Window;
use yapocis::lib::Fft;
use yapocis::Device;

const EX: &str = "fft_library_demo";

fn tone_sum(sample_rate: f32, duration: f32, parts: &[(f32, f32)]) -> (Vec<f32>, Array1<f32>) {
    let n = (sample_rate * duration) as usize;
    let t = common::linspace_f32(0.0, duration, n);
    let tp = 2.0 * std::f32::consts::PI;
    let s = Array1::from_iter(t.iter().map(|&t| {
        parts
            .iter()
            .map(|(a, f)| a * (tp * f * t).sin())
            .sum::<f32>()
    }));
    (t, s)
}

fn spectrum_axes(fig: &Figure, rows: usize, i: usize, xmax: f64, p: &[f64]) -> Axes {
    let (lo, hi) = p
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    Axes::subplot(fig, rows, 1, i, (0.0, xmax), (lo - 5.0, hi + 5.0))
}

fn demo_basic_analysis(device: &Device, fft: &Fft) -> yapocis::Result<()> {
    println!(
        "\n{}\nDemo 1: Basic Spectral Analysis\n{}",
        common::rule(60),
        common::rule(60)
    );
    let sr = 8000.0;
    let (_, signal) = tone_sum(sr, 0.5, &[(1.0, 440.0), (0.5, 880.0)]);
    let (freqs, power) = fft.analyze_signal(&signal.view(), sr as f64, Window::Hann, true, true)?;
    let (pf, pp) = fft.find_peaks(&signal.view(), sr as f64, 2, Window::Hann)?;
    println!("Dominant frequencies found:");
    for (i, (f, p)) in pf.iter().zip(&pp).enumerate() {
        println!("  Peak {}: {f:.1} Hz ({p:.1} dB)", i + 1);
    }
    let mut fig = Figure::new(device, 1000, 500)?;
    let p: Vec<f64> = power.iter().map(|&v| v as f64).collect();
    let mut ax = spectrum_axes(&fig, 1, 0, 2000.0, &p);
    ax.line(&mut fig, &freqs.to_vec(), &p, &Line::solid(BLUE, 1.0), None)?;
    for &f in &pf {
        ax.vline(&mut fig, f, &Line::dashed(with_alpha(RED, 0.5), 1.5))?;
    }
    ax.decorate(
        &mut fig,
        "Power Spectrum - Library API Demo",
        "Frequency (Hz)",
        "Power (dB)",
    )?;
    println!();
    fig.save(&common::out_dir(EX).join("fft_library_demo_basic.png"))
}

fn demo_spectrogram(device: &Device, fft: &Fft) -> yapocis::Result<()> {
    println!(
        "\n{}\nDemo 2: Spectrogram Generation\n{}",
        common::rule(60),
        common::rule(60)
    );
    let (sr, duration, f0, f1) = (8000.0f32, 2.0f32, 200.0f32, 1000.0f32);
    let n = (sr * duration) as usize;
    let t = common::linspace_f32(0.0, duration, n);
    let k = (f1 - f0) / duration;
    let signal = Array1::from_iter(
        t.iter()
            .map(|&t| (2.0 * std::f32::consts::PI * (f0 * t + 0.5 * k * t * t)).sin()),
    );
    let (times, freqs, spec) = fft.spectrogram(
        &signal.view(),
        sr as f64,
        512,
        Some(128),
        Window::Hann,
        true,
    )?;
    println!("Spectrogram shape: ({}, {})", spec.nrows(), spec.ncols());
    println!(
        "Time range: {:.3} to {:.3} seconds",
        times[0],
        times[times.len() - 1]
    );
    println!(
        "Frequency range: {:.1} to {:.1} Hz",
        freqs[0],
        freqs[freqs.len() - 1]
    );
    let keep = freqs.iter().take_while(|&&f| f <= 1500.0).count().max(1);
    let mut fig = Figure::new(device, 1200, 600)?;
    let mut ax = Axes::subplot(
        &fig,
        1,
        1,
        0,
        (times[0] as f64, times[times.len() - 1] as f64),
        (freqs[0], freqs[keep - 1]),
    );
    ax.heatmap(
        &mut fig,
        &spec.slice(ndarray::s![.., ..keep]).t().to_owned(),
    )?;
    let ct = common::plot::linspace(0.0, duration as f64, 100);
    let cf: Vec<f64> = ct
        .iter()
        .map(|&t| f0 as f64 + (f1 - f0) as f64 * t / duration as f64)
        .collect();
    ax.line(
        &mut fig,
        &ct,
        &cf,
        &Line::dashed(RED, 2.0),
        Some("Expected"),
    )?;
    ax.decorate(
        &mut fig,
        "Spectrogram - Library API Demo",
        "Time (s)",
        "Frequency (Hz)",
    )?;
    ax.legend(&mut fig, false)?;
    println!();
    fig.save(&common::out_dir(EX).join("fft_library_demo_spectrogram.png"))
}

fn demo_bandpass_filter(device: &Device, fft: &Fft) -> yapocis::Result<()> {
    println!(
        "\n{}\nDemo 3: Bandpass Filtering\n{}",
        common::rule(60),
        common::rule(60)
    );
    let sr = 1000.0;
    let (_, signal) = tone_sum(
        sr,
        1.0,
        &[(1.0, 50.0), (1.0, 100.0), (1.0, 150.0), (1.0, 300.0)],
    );
    let filtered = fft.bandpass_filter(&signal.view(), 80.0, 200.0, sr as f64)?;
    let (fo, po) = fft.analyze_signal(&signal.view(), sr as f64, Window::Hann, true, true)?;
    let (ff, pf) = fft.analyze_signal(&filtered.view(), sr as f64, Window::Hann, true, true)?;
    println!("Original signal frequency content:");
    let (mut peaks, _) = fft.find_peaks(&signal.view(), sr as f64, 4, Window::Hann)?;
    peaks.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for f in peaks {
        println!("  {f:.1} Hz");
    }
    println!("\nFiltered signal frequency content (80-200 Hz passband):");
    let (mut peaks, _) = fft.find_peaks(&filtered.view(), sr as f64, 4, Window::Hann)?;
    peaks.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for f in peaks {
        println!("  {f:.1} Hz");
    }
    let mut fig = Figure::new(device, 1200, 800)?;
    for (i, (f, p, title, color)) in [
        (fo, po, "Original Signal Spectrum", BLUE),
        (ff, pf, "Filtered Signal Spectrum (80-200 Hz)", RED),
    ]
    .into_iter()
    .enumerate()
    {
        let p: Vec<f64> = p.iter().map(|&v| v as f64).collect();
        let mut ax = spectrum_axes(&fig, 2, i, 400.0, &p);
        ax.vspan(
            &mut fig,
            80.0,
            200.0,
            with_alpha(GREEN, 0.2),
            Some("Passband"),
        )?;
        ax.line(&mut fig, &f.to_vec(), &p, &Line::solid(color, 1.0), None)?;
        ax.decorate(&mut fig, title, "Frequency (Hz)", "Power (dB)")?;
        ax.legend(&mut fig, false)?;
    }
    println!();
    fig.save(&common::out_dir(EX).join("fft_library_demo_filter.png"))
}

fn demo_signal_metrics() {
    println!(
        "\n{}\nDemo 4: Signal Quality Metrics\n{}",
        common::rule(60),
        common::rule(60)
    );
    let sr = 1000.0;
    let (_, clean) = tone_sum(sr, 1.0, &[(1.0, 100.0)]);
    let mut rng = common::Rng::new(5);
    let noise: Vec<f32> = (0..clean.len()).map(|_| 0.1 * rng.normal()).collect();
    let noisy: Vec<f32> = clean.iter().zip(&noise).map(|(a, b)| a + b).collect();
    let clean = clean.to_vec();
    println!("Clean signal:");
    println!("  Energy: {:.2}", Fft::energy(&clean));
    println!("  RMS: {:.4}", Fft::rms(&clean));
    println!("\nNoise:");
    println!("  Energy: {:.2}", Fft::energy(&noise));
    println!("\nNoisy signal:");
    println!("  RMS: {:.4}", Fft::rms(&noisy));
    println!(
        "\nSignal-to-Noise Ratio: {:.2} dB",
        Fft::snr(&clean, &noise)
    );
}

fn demo_coherence(device: &Device, fft: &Fft) -> yapocis::Result<()> {
    println!(
        "\n{}\nDemo 5: Coherence Analysis\n{}",
        common::rule(60),
        common::rule(60)
    );
    let n = 1000;
    let mut rng = common::Rng::new(6);
    let s1: Array1<f32> = Array1::from_iter((0..n).map(|_| rng.normal()));
    let s2: Array1<f32> = Array1::from_iter(s1.iter().map(|v| 0.9 * v + 0.1 * rng.normal()));
    let s3: Array1<f32> = Array1::from_iter((0..n).map(|_| rng.normal()));
    let (freqs, high) = fft.coherence(&s1.view(), &s2.view(), Window::Hann)?;
    let (_, low) = fft.coherence(&s1.view(), &s3.view(), Window::Hann)?;
    println!(
        "Coherence between correlated signals: {:.3}",
        high.mean().unwrap()
    );
    println!(
        "Coherence between uncorrelated signals: {:.3}",
        low.mean().unwrap()
    );
    let fx = freqs.to_vec();
    let (fmin, fmax) = fx
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let mut fig = Figure::new(device, 1000, 500)?;
    let mut ax = Axes::subplot(&fig, 1, 1, 0, (fmin, fmax), (0.0, 1.1));
    // the coherence frequencies are fftfreq order; plot in increasing order
    let mut order: Vec<usize> = (0..fx.len()).collect();
    order.sort_by(|&a, &b| fx[a].partial_cmp(&fx[b]).unwrap());
    let xs: Vec<f64> = order.iter().map(|&i| fx[i]).collect();
    for (c, label, color) in [
        (&high, "Correlated signals", GREEN),
        (&low, "Uncorrelated signals", RED),
    ] {
        let ys: Vec<f64> = order.iter().map(|&i| c[i] as f64).collect();
        ax.line(
            &mut fig,
            &xs,
            &ys,
            &Line::solid(with_alpha(color, 0.7), 1.0),
            Some(label),
        )?;
    }
    ax.decorate(
        &mut fig,
        "Coherence Analysis",
        "Normalized Frequency",
        "Coherence",
    )?;
    ax.legend(&mut fig, false)?;
    println!();
    fig.save(&common::out_dir(EX).join("fft_library_demo_coherence.png"))
}

fn main() -> yapocis::Result<()> {
    println!(
        "{}\nFFT Library API Demo\n{}",
        common::rule(60),
        common::rule(60)
    );
    println!("\nThis demo shows how the high-level FFT library makes");
    println!("signal processing simple and intuitive!");
    let device = common::device();
    let fft = Fft::load(&device)?;
    demo_basic_analysis(&device, &fft)?;
    demo_spectrogram(&device, &fft)?;
    demo_bandpass_filter(&device, &fft)?;
    demo_signal_metrics();
    demo_coherence(&device, &fft)?;
    println!(
        "\n{}\nAll demos complete!\n{}",
        common::rule(60),
        common::rule(60)
    );
    println!("\nKey advantages of the library API:");
    for s in [
        "Simple, intuitive method names",
        "Sensible defaults (Hann window, dB scale, etc.)",
        "Automatic array conversions",
        "Comprehensive signal analysis tools",
        "One-line spectrograms and filters",
    ] {
        println!("  ✓ {s}");
    }
    println!("{}", common::rule(60));
    Ok(())
}
