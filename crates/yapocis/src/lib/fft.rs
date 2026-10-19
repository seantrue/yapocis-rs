//! `yapocis.lib.fft`: 1-D FFTs, batches, windows and power spectra.

use crate::host;
use crate::interfaces::fft;
use crate::rt::CallOptions;
use crate::{Device, Error, Result};
use ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayView2, Axis};
use num_complex::Complex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Window {
    Rectangular = 0,
    Hann = 1,
    Hamming = 2,
    Blackman = 3,
}

pub struct Fft {
    pub kernels: fft::Fft,
}

impl Fft {
    pub fn load(device: &Device) -> Result<Fft> {
        Ok(Fft {
            kernels: fft::Fft::load(device)?,
        })
    }

    /// `FFT.forward`/`inverse`: the optimised path uses a local size equal to
    /// the FFT size when it is at most 256; larger sizes let the driver choose.
    fn fft1d(
        &self,
        signal: &ArrayView1<'_, Complex<f32>>,
        direction: i32,
    ) -> Result<Array1<Complex<f32>>> {
        let n = signal.len();
        let opts = CallOptions {
            global: Some(vec![n]),
            local: if n <= 256 { Some(vec![n]) } else { None },
            ..Default::default()
        };
        let out: ArrayD<Complex<f32>> = host::one_output(self.kernels.fft1d_with(
            &opts,
            &signal.to_owned().into_dyn().view(),
            direction,
        )?)?;
        Ok(out.into_dimensionality()?)
    }

    pub fn forward(&self, signal: &ArrayView1<'_, Complex<f32>>) -> Result<Array1<Complex<f32>>> {
        self.fft1d(signal, 1)
    }

    pub fn inverse(&self, spectrum: &ArrayView1<'_, Complex<f32>>) -> Result<Array1<Complex<f32>>> {
        self.fft1d(spectrum, -1)
    }

    /// `FFT.batch_forward`/`batch_inverse`: one launch for `(batch, fft_size)`
    /// signals; `fft_size` must be a power of two no larger than 256.
    fn batch(
        &self,
        signals: &ArrayView2<'_, Complex<f32>>,
        direction: i32,
    ) -> Result<Array2<Complex<f32>>> {
        let (batch_size, fft_size) = signals.dim();
        if !fft_size.is_power_of_two() || fft_size > 256 {
            return Err(Error::Invalid(format!(
                "batch FFT size {fft_size} must be a power of two <= 256"
            )));
        }
        let opts = CallOptions {
            global: Some(vec![batch_size * fft_size]),
            local: Some(vec![fft_size]),
            ..Default::default()
        };
        let input = signals.to_owned().into_dyn();
        let out: ArrayD<Complex<f32>> = host::one_output(self.kernels.fft1d_batch_with(
            &opts,
            &input.view(),
            batch_size as i32,
            fft_size as i32,
            direction,
        )?)?;
        Ok(out.into_dimensionality()?)
    }

    pub fn batch_forward(
        &self,
        signals: &ArrayView2<'_, Complex<f32>>,
    ) -> Result<Array2<Complex<f32>>> {
        self.batch(signals, 1)
    }

    pub fn batch_inverse(
        &self,
        spectra: &ArrayView2<'_, Complex<f32>>,
    ) -> Result<Array2<Complex<f32>>> {
        self.batch(spectra, -1)
    }

    /// `FFT.apply_window(signal, window_type)`.
    pub fn apply_window(
        &self,
        signal: &ArrayView1<'_, f32>,
        window: Window,
    ) -> Result<Array1<f32>> {
        let out = self
            .kernels
            .window(&signal.to_owned().into_dyn().view(), window as i32)?;
        Ok(out.into_dimensionality()?)
    }

    /// `FFT.compute_power_spectrum(fft_result)`: magnitude squared.
    pub fn power_spectrum(&self, spectrum: &ArrayView1<'_, Complex<f32>>) -> Result<Array1<f32>> {
        let out = self
            .kernels
            .power_spectrum(&spectrum.to_owned().into_dyn().view())?;
        Ok(out.into_dimensionality()?)
    }

    /// `FFT.analyze_signal(signal, sample_rate, window_type, return_db, positive_only)`:
    /// returns `(freqs, power)`.
    pub fn analyze_signal(
        &self,
        signal: &ArrayView1<'_, f32>,
        sample_rate: f64,
        window: Window,
        return_db: bool,
        positive_only: bool,
    ) -> Result<(Array1<f64>, Array1<f32>)> {
        let n = signal.len();
        let windowed = self.apply_window(signal, window)?;
        let complex: Array1<Complex<f32>> = windowed.mapv(|v| Complex::new(v, 0.0));
        let spectrum = self.forward(&complex.view())?;
        let mut power = self.power_spectrum(&spectrum.view())?;
        if return_db {
            power.mapv_inplace(|p| 10.0 * (p + 1e-10).log10());
        }
        let mut freqs = fftfreq(n, 1.0 / sample_rate);
        if positive_only {
            let keep: Vec<usize> = (0..n).filter(|&i| freqs[i] >= 0.0).collect();
            freqs = freqs.select(Axis(0), &keep);
            power = power.select(Axis(0), &keep);
        }
        Ok((freqs, power))
    }

    /// `FFT.find_peaks`: the `n_peaks` strongest positive-frequency bins, strongest first.
    pub fn find_peaks(
        &self,
        signal: &ArrayView1<'_, f32>,
        sample_rate: f64,
        n_peaks: usize,
        window: Window,
    ) -> Result<(Vec<f64>, Vec<f32>)> {
        let (freqs, power) = self.analyze_signal(signal, sample_rate, window, true, true)?;
        let mut idx: Vec<usize> = (0..power.len()).collect();
        idx.sort_by(|&a, &b| {
            power[b]
                .partial_cmp(&power[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(n_peaks);
        Ok((
            idx.iter().map(|&i| freqs[i]).collect(),
            idx.iter().map(|&i| power[i]).collect(),
        ))
    }

    /// `FFT.spectrogram(signal, sample_rate, window_size=512, hop_size=None, window_type, return_db)`:
    /// returns `(times, freqs, spectrogram)` with `spectrogram[time, freq]` over
    /// the first `window_size / 2` bins. Ten or more windows of at most 256
    /// samples go through the batch FFT, as in Python.
    pub fn spectrogram(
        &self,
        signal: &ArrayView1<'_, f32>,
        sample_rate: f64,
        window_size: usize,
        hop_size: Option<usize>,
        window: Window,
        return_db: bool,
    ) -> Result<(Array1<f32>, Array1<f64>, Array2<f32>)> {
        let hop_size = hop_size.unwrap_or(window_size / 4).max(1);
        let n_samples = signal.len();
        if n_samples < window_size {
            return Err(Error::Invalid(format!(
                "signal of {n_samples} samples is shorter than the window ({window_size})"
            )));
        }
        let n_windows = (n_samples - window_size) / hop_size + 1;
        let freqs = fftfreq(window_size, 1.0 / sample_rate)
            .slice(ndarray::s![..window_size / 2])
            .to_owned();
        let n_freqs = freqs.len();
        let mut spec = Array2::<f32>::zeros((n_windows, n_freqs));
        let mut times = Array1::<f32>::zeros(n_windows);
        let use_batch = n_windows >= 10 && window_size <= 256;
        if use_batch {
            let mut windowed = Array2::<f32>::zeros((n_windows, window_size));
            for i in 0..n_windows {
                let start = i * hop_size;
                let end = start + window_size;
                times[i] = ((start + end) as f64 / (2.0 * sample_rate)) as f32;
                let w = self.apply_window(&signal.slice(ndarray::s![start..end]), window)?;
                windowed.row_mut(i).assign(&w);
            }
            let complex = windowed.mapv(|v| Complex::new(v, 0.0));
            let results = self.batch_forward(&complex.view())?;
            for i in 0..n_windows {
                let power = self.power_spectrum(&results.row(i))?;
                spec.row_mut(i).assign(&power.slice(ndarray::s![..n_freqs]));
            }
        } else {
            for i in 0..n_windows {
                let start = i * hop_size;
                let end = start + window_size;
                times[i] = ((start + end) as f64 / (2.0 * sample_rate)) as f32;
                let w = self.apply_window(&signal.slice(ndarray::s![start..end]), window)?;
                let complex = w.mapv(|v| Complex::new(v, 0.0));
                let result = self.forward(&complex.view())?;
                let power = self.power_spectrum(&result.view())?;
                spec.row_mut(i).assign(&power.slice(ndarray::s![..n_freqs]));
            }
        }
        if return_db {
            spec.mapv_inplace(|p| 10.0 * (p + 1e-10).log10());
        }
        Ok((times, freqs, spec))
    }

    /// `FFT.estimate_frequency(signal, sample_rate, window_type)`: the dominant frequency.
    pub fn estimate_frequency(
        &self,
        signal: &ArrayView1<'_, f32>,
        sample_rate: f64,
        window: Window,
    ) -> Result<f64> {
        let (peaks, _) = self.find_peaks(signal, sample_rate, 1, window)?;
        peaks
            .first()
            .copied()
            .ok_or_else(|| Error::Invalid("empty signal".into()))
    }

    /// `FFT.bandpass_filter(signal, low_freq, high_freq, sample_rate)`: FFT,
    /// zero bins outside `[low, high]` in |frequency|, inverse FFT, real part.
    pub fn bandpass_filter(
        &self,
        signal: &ArrayView1<'_, f32>,
        low_freq: f64,
        high_freq: f64,
        sample_rate: f64,
    ) -> Result<Array1<f32>> {
        let n = signal.len();
        let complex: Array1<Complex<f32>> = signal.mapv(|v| Complex::new(v, 0.0));
        let spectrum = self.forward(&complex.view())?;
        let freqs = fftfreq(n, 1.0 / sample_rate);
        let filtered: Array1<Complex<f32>> = Array1::from_iter((0..n).map(|i| {
            let f = freqs[i].abs();
            if f >= low_freq && f <= high_freq {
                spectrum[i]
            } else {
                Complex::new(0.0, 0.0)
            }
        }));
        let result = self.inverse(&filtered.view())?;
        Ok(result.mapv(|c| c.re))
    }

    /// `FFT.coherence(signal1, signal2, window_type)`: magnitude-squared
    /// coherence over the positive normalised frequencies.
    pub fn coherence(
        &self,
        signal1: &ArrayView1<'_, f32>,
        signal2: &ArrayView1<'_, f32>,
        window: Window,
    ) -> Result<(Array1<f64>, Array1<f32>)> {
        let n = signal1.len().min(signal2.len());
        let w1 = self.apply_window(&signal1.slice(ndarray::s![..n]), window)?;
        let w2 = self.apply_window(&signal2.slice(ndarray::s![..n]), window)?;
        let f1 = self.forward(&w1.mapv(|v| Complex::new(v, 0.0)).view())?;
        let f2 = self.forward(&w2.mapv(|v| Complex::new(v, 0.0)).view())?;
        let coh: Array1<f32> = Array1::from_iter((0..n).map(|i| {
            let cross = f1[i] * f2[i].conj();
            let auto1 = f1[i].norm_sqr();
            let auto2 = f2[i].norm_sqr();
            cross.norm_sqr() / (auto1 * auto2 + 1e-10)
        }));
        let freqs = fftfreq(n, 1.0);
        let keep: Vec<usize> = (0..n).filter(|&i| freqs[i] >= 0.0).collect();
        Ok((freqs.select(Axis(0), &keep), coh.select(Axis(0), &keep)))
    }

    /// `FFT.energy(signal)`: sum of squared magnitudes.
    pub fn energy(signal: &[f32]) -> f64 {
        signal.iter().map(|&v| (v as f64) * (v as f64)).sum()
    }

    /// `FFT.energy` for complex signals.
    pub fn energy_complex(signal: &[Complex<f32>]) -> f64 {
        signal.iter().map(|c| c.norm_sqr() as f64).sum()
    }

    /// `FFT.rms(signal)`.
    pub fn rms(signal: &[f32]) -> f64 {
        (Self::energy(signal) / signal.len().max(1) as f64).sqrt()
    }

    /// `FFT.snr(signal, noise)` in dB.
    pub fn snr(signal: &[f32], noise: &[f32]) -> f64 {
        10.0 * (Self::energy(signal) / (Self::energy(noise) + 1e-10)).log10()
    }
}

/// `numpy.fft.fftfreq(n, d)`.
pub fn fftfreq(n: usize, d: f64) -> Array1<f64> {
    let val = 1.0 / (n as f64 * d);
    let half = (n - 1) / 2 + 1;
    Array1::from_iter((0..n).map(|i| {
        let k = if i < half {
            i as i64
        } else {
            i as i64 - n as i64
        };
        k as f64 * val
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fftfreq_matches_numpy() {
        let f = fftfreq(8, 0.125);
        assert_eq!(f.to_vec(), vec![0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0]);
        let f = fftfreq(5, 1.0);
        assert_eq!(f.to_vec(), vec![0.0, 0.2, 0.4, -0.4, -0.2]);
    }
}
