//! `examples/fft_quickstart.py`: analyse a 440 Hz + 880 Hz signal with the FFT
//! library, find the two peaks, and plot the time signal and power spectrum.
//!
//! The matplotlib figure is redrawn with the GPU plotting helper in
//! `examples/common/plot.rs` (same content, not matplotlib's layout); the
//! interactive `plt.show()` is dropped.
//!
//! cargo run -p yapocis --example fft_quickstart --features metal --release

mod common;

use common::plot::{Axes, Figure, Line, BLUE, GREEN, RED};
use ndarray::Array1;
use yapocis::lib::fft::Window;
use yapocis::lib::Fft;

fn main() -> yapocis::Result<()> {
    println!(
        "{}\nFFT Quickstart Example\n{}\n",
        common::rule(60),
        common::rule(60)
    );
    let device = common::device();
    let fft = Fft::load(&device)?;
    println!("FFT library initialized!");

    let sample_rate = 8000.0;
    let duration = 0.5;
    let n_samples = (sample_rate * duration) as usize;
    let t = common::linspace_f32(0.0, duration as f32, n_samples);
    let two_pi = 2.0 * std::f32::consts::PI;
    let signal = Array1::from_iter(
        t.iter()
            .map(|&t| (two_pi * 440.0 * t).sin() + 0.5 * (two_pi * 880.0 * t).sin()),
    );
    println!("Created signal: {n_samples} samples at {sample_rate} Hz");
    println!("Frequencies: 440 Hz and 880 Hz");

    let (freqs, power_db) =
        fft.analyze_signal(&signal.view(), sample_rate, Window::Hann, true, true)?;
    let (peak_freqs, peak_powers) = fft.find_peaks(&signal.view(), sample_rate, 2, Window::Hann)?;

    let mut fig = Figure::new(&device, 1200, 800)?;
    let mut ax1 = Axes::subplot(&fig, 2, 1, 0, (0.0, t[499] as f64), (-1.6, 1.6));
    let ts: Vec<f64> = t[..500].iter().map(|&v| v as f64).collect();
    let ys: Vec<f64> = signal.iter().take(500).map(|&v| v as f64).collect();
    ax1.line(&mut fig, &ts, &ys, &Line::solid(BLUE, 1.0), None)?;
    ax1.decorate(
        &mut fig,
        "Time-Domain Signal: 440 Hz + 880 Hz",
        "Time (s)",
        "Amplitude",
    )?;

    let keep: Vec<usize> = (0..freqs.len()).filter(|&i| freqs[i] <= 2000.0).collect();
    let fx: Vec<f64> = keep.iter().map(|&i| freqs[i]).collect();
    let fy: Vec<f64> = keep.iter().map(|&i| power_db[i] as f64).collect();
    let (lo, hi) = fy
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let mut ax2 = Axes::subplot(&fig, 2, 1, 1, (0.0, 2000.0), (lo - 5.0, hi + 5.0));
    ax2.line(
        &mut fig,
        &fx,
        &fy,
        &Line::solid(RED, 1.5),
        Some("Power Spectrum"),
    )?;
    for &f in &peak_freqs {
        ax2.vline(
            &mut fig,
            f,
            &Line::dashed(common::plot::with_alpha(GREEN, 0.6), 1.5),
        )?;
    }
    ax2.add_legend("Detected Peaks", GREEN);
    ax2.decorate(
        &mut fig,
        "Power Spectrum (Hann Window)",
        "Frequency (Hz)",
        "Power (dB)",
    )?;
    ax2.legend(&mut fig, false)?;
    let out = common::out_dir("fft_quickstart").join("fft_quickstart.png");
    println!();
    fig.save(&out)?;

    println!("\nDetected frequency peaks:");
    for (i, (f, p)) in peak_freqs.iter().zip(&peak_powers).enumerate() {
        println!("  Peak {}: {f:.1} Hz ({p:.1} dB)", i + 1);
    }
    println!("\n{}\nDone!\n{}", common::rule(60), common::rule(60));
    Ok(())
}
