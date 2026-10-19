//! `examples/fft_chirp_analysis.py`: spectrogram of a 100 → 1000 Hz linear
//! chirp, with the expected frequency track and spectra at three instants.
//!
//! The matplotlib figure (signal, `imshow` spectrogram with viridis, slices) is
//! redrawn with `examples/common/plot.rs`; the colour bar and `plt.show()` are
//! dropped. Note: the library's batch-FFT spectrogram path is known to be
//! nondeterministic on Apple backends in the Python package too
//! (claude-debug/BATCH_FFT_CROSSBACKEND_STATUS.md), so a run may show stray bins.
//!
//! cargo run -p yapocis --example fft_chirp_analysis --features metal --release

mod common;

use common::plot::{with_alpha, Axes, Figure, Line, BLUE, GREEN, RED};
use ndarray::Array1;
use yapocis::lib::fft::Window;
use yapocis::lib::Fft;

pub fn create_chirp_signal(
    duration: f32,
    sample_rate: f32,
    f_start: f32,
    f_end: f32,
) -> (Vec<f32>, Array1<f32>) {
    let n_samples = (sample_rate * duration) as usize;
    let t = common::linspace_f32(0.0, duration, n_samples);
    let k = (f_end - f_start) / duration;
    let signal = Array1::from_iter(
        t.iter()
            .map(|&t| (2.0 * std::f32::consts::PI * (f_start * t + 0.5 * k * t * t)).sin()),
    );
    (t, signal)
}

fn main() -> yapocis::Result<()> {
    println!(
        "{}\nFFT Chirp Signal Analysis Example\n{}\n",
        common::rule(70),
        common::rule(70)
    );
    let device = common::device();
    println!("Initializing FFT library...");
    let fft = Fft::load(&device)?;

    println!("\nCreating chirp signal...");
    let (sample_rate, duration, f_start, f_end) = (8000.0f32, 2.0f32, 100.0f32, 1000.0f32);
    let (t, signal) = create_chirp_signal(duration, sample_rate, f_start, f_end);
    println!("Signal: {} samples, {duration}s", signal.len());
    println!("Frequency sweep: {f_start} Hz -> {f_end} Hz");

    println!("\nGenerating spectrogram...");
    let (times, freqs, spectra) = fft.spectrogram(
        &signal.view(),
        sample_rate as f64,
        512,
        Some(128),
        Window::Hann,
        true,
    )?;
    println!(
        "Spectrogram generated: {} time windows, {} frequency bins",
        spectra.nrows(),
        spectra.ncols()
    );

    let mut fig = Figure::new(&device, 1400, 1000)?;
    let mut ax1 = Axes::subplot(&fig, 3, 1, 0, (0.0, duration as f64), (-1.2, 1.2));
    let ts: Vec<f64> = t.iter().map(|&v| v as f64).collect();
    let ys: Vec<f64> = signal.iter().map(|&v| v as f64).collect();
    ax1.line(
        &mut fig,
        &ts,
        &ys,
        &Line::solid(with_alpha(BLUE, 0.7), 1.0),
        None,
    )?;
    ax1.decorate(
        &mut fig,
        &format!("Chirp Signal: {f_start} Hz -> {f_end} Hz"),
        "Time (s)",
        "Amplitude",
    )?;

    // Spectrogram: keep bins up to 1500 Hz (the Python axes' ylim)
    let keep = freqs.iter().take_while(|&&f| f <= 1500.0).count().max(1);
    let shown = spectra.slice(ndarray::s![.., ..keep]).t().to_owned();
    let mut ax2 = Axes::subplot(
        &fig,
        3,
        1,
        1,
        (times[0] as f64, *times.last().unwrap() as f64),
        (freqs[0], freqs[keep - 1]),
    );
    ax2.heatmap(&mut fig, &shown)?;
    let ct = common::plot::linspace(0.0, duration as f64, 100);
    let cf: Vec<f64> = ct
        .iter()
        .map(|&t| f_start as f64 + (f_end - f_start) as f64 * t / duration as f64)
        .collect();
    ax2.xlim = (0.0, duration as f64);
    ax2.line(
        &mut fig,
        &ct,
        &cf,
        &Line::dashed(RED, 2.0),
        Some("Expected frequency"),
    )?;
    ax2.decorate(
        &mut fig,
        "Spectrogram (Time-Frequency Analysis)",
        "Time (s)",
        "Frequency (Hz)",
    )?;
    ax2.legend(&mut fig, true)?;

    let lo = spectra.iter().cloned().fold(f32::MAX, f32::min) as f64;
    let hi = spectra.iter().cloned().fold(f32::MIN, f32::max) as f64;
    let mut ax3 = Axes::subplot(&fig, 3, 1, 2, (0.0, 1500.0), (lo, hi + 5.0));
    for (tp, color) in [(0.5f32, BLUE), (1.0, GREEN), (1.5, RED)] {
        let idx = (0..times.len())
            .min_by(|&a, &b| {
                (times[a] - tp)
                    .abs()
                    .partial_cmp(&(times[b] - tp).abs())
                    .unwrap()
            })
            .unwrap();
        let actual = times[idx];
        let expected = f_start + (f_end - f_start) * actual / duration;
        let row: Vec<f64> = spectra.row(idx).iter().map(|&v| v as f64).collect();
        ax3.line(
            &mut fig,
            &freqs.to_vec(),
            &row,
            &Line::solid(with_alpha(color, 0.7), 1.5),
            Some(&format!("t={actual:.2}s (~{expected:.0} Hz)")),
        )?;
        ax3.vline(
            &mut fig,
            expected as f64,
            &Line::dotted(with_alpha(color, 0.5), 1.0),
        )?;
    }
    ax3.decorate(
        &mut fig,
        "Power Spectra at Different Time Points",
        "Frequency (Hz)",
        "Power (dB)",
    )?;
    ax3.legend(&mut fig, false)?;

    let out = common::out_dir("fft_chirp_analysis").join("fft_chirp_analysis.png");
    println!("\nSaving plot to {}...", out.display());
    fig.save(&out)?;
    println!("\nAnalysis complete!");
    println!(
        "The spectrogram clearly shows the frequency increasing from {f_start} Hz to {f_end} Hz"
    );
    println!("over the {duration:.1} second duration.");
    println!("\n{}\nDone!\n{}", common::rule(70), common::rule(70));
    Ok(())
}
