//! Time-delay estimation between two channels.
//!
//! [`gcc_phat`] is the generalised cross-correlation with the phase transform:
//! the cross-spectrum of the two signals, each bin divided by its own
//! magnitude, transformed back to a correlation over lags. For sound with
//! energy across the whole band, a pure delay between the channels becomes one
//! sharp peak at that lag; [`gcc_phat`] says what happens on other sound.

use std::f32::consts::PI;
use std::fmt;

use crate::fft::{self, Complex};

/// Narrowing steps of the fractional search, and points per step. Each step
/// keeps a quarter of the interval, so eight steps resolve the peak to about
/// `2 / 4^8` frame.
const REFINE_STEPS: usize = 8;
const REFINE_POINTS: usize = 9;

/// The delay between two channels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Delay {
    /// Frames by which `b` lags `a`, fractional. Negative when `b` leads.
    pub frames: f32,
    /// The weighted correlation at the peak: 1.0 for two identical signals.
    pub peak: f32,
}

/// Why [`gcc_phat`] refused its input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelayError {
    /// The two channels have different lengths.
    LengthMismatch { a: usize, b: usize },
    /// The channels are not longer than the largest lag searched.
    MaxLagTooLarge { len: usize, max_lag: usize },
    /// A sample is NaN or infinite.
    NonFinite,
    /// Every bin of the cross-spectrum is zero, as when either channel is
    /// silent.
    NoSignal,
}

impl fmt::Display for DelayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DelayError::LengthMismatch { a, b } => {
                write!(f, "channel lengths differ: {a} and {b} frames")
            }
            DelayError::MaxLagTooLarge { len, max_lag } => {
                write!(f, "max_lag {max_lag} is not below the channel length {len}")
            }
            DelayError::NonFinite => write!(f, "a sample is not finite"),
            DelayError::NoSignal => write!(f, "the cross-spectrum is zero in every bin"),
        }
    }
}

impl std::error::Error for DelayError {}

/// Estimates the delay of `b` relative to `a`, searched within
/// `-max_lag..=max_lag` frames.
///
/// Both channels must have the same length, longer than `max_lag`, and every
/// sample must be finite. The result is the lag of the largest weighted
/// correlation within the range, refined between frames. When the true delay
/// is outside the range, the result is still within it, often at its bound.
///
/// # Accuracy
///
/// The estimate is accurate on sound that has energy in every part of the
/// band, up to half the sample rate: this crate's tests hold it within 0.05
/// frame on such signals. The phase transform gives every frequency bin the
/// same weight, so a bin the sound barely reaches counts as much as one it
/// fills, and what that bin holds need not carry the delay. On sound that
/// leaves part of the band empty, such as a few tones or sound confined to
/// the lower part of the band, the estimate can be wrong by anything from a
/// small fraction of a frame to whole frames, on either side of the true
/// delay, and `peak` can be as high as it is for a right estimate. Such
/// input is not refused: whether a signal fills the band is for the caller
/// to know.
pub fn gcc_phat(a: &[f32], b: &[f32], max_lag: usize) -> Result<Delay, DelayError> {
    if a.len() != b.len() {
        return Err(DelayError::LengthMismatch {
            a: a.len(),
            b: b.len(),
        });
    }
    let len = a.len();
    if len <= max_lag {
        return Err(DelayError::MaxLagTooLarge { len, max_lag });
    }
    if !a.iter().chain(b).all(|x| x.is_finite()) {
        return Err(DelayError::NonFinite);
    }

    // Padded so that no lag within the range wraps around the circular
    // correlation.
    let n = (len + max_lag + 1).next_power_of_two();
    let spectrum_a = spectrum(a, n);
    let spectrum_b = spectrum(b, n);

    // The phase transform: each bin of conj(A) B divided by its magnitude,
    // formed as the product of the two bins' unit phasors so that no product
    // of magnitudes can overflow or underflow. A bin of zero magnitude in
    // either spectrum carries no phase and gets no weight.
    let mut weighted = Vec::with_capacity(n);
    let mut weighted_bins = 0usize;
    for (x, y) in spectrum_a.iter().zip(&spectrum_b) {
        match (unit_phasor(*x), unit_phasor(*y)) {
            (Some(x), Some(y)) => {
                weighted.push(x.conj().mul(y));
                weighted_bins += 1;
            }
            _ => weighted.push(Complex::ZERO),
        }
    }
    if weighted_bins == 0 {
        return Err(DelayError::NoSignal);
    }

    let mut correlation = weighted.clone();
    fft::inverse(&mut correlation);
    let lag_value = |lag: isize| correlation[lag.rem_euclid(n as isize) as usize].re;
    let bound = max_lag as isize;
    let mut best = 0isize;
    for lag in -bound..=bound {
        if lag_value(lag) > lag_value(best) {
            best = lag;
        }
    }

    let (frames, value) = refine(&weighted, best as f32, max_lag as f32);
    Ok(Delay {
        frames,
        peak: value / weighted_bins as f32,
    })
}

/// The transform of `signal` divided by its largest magnitude and zero-padded
/// to `n` frames. The phase transform ignores each channel's scale, and at
/// unit peak no sum in the transform can overflow and no sample of a quiet
/// channel loses precision to the subnormal range.
fn spectrum(signal: &[f32], n: usize) -> Vec<Complex> {
    let mut data = vec![Complex::ZERO; n];
    let peak = signal.iter().fold(0.0f32, |peak, x| peak.max(x.abs()));
    if peak > 0.0 {
        for (slot, &x) in data.iter_mut().zip(signal) {
            slot.re = x / peak;
        }
    }
    fft::forward(&mut data);
    data
}

/// `z` divided by its magnitude, or `None` when the magnitude is zero.
/// Dividing by the magnitude, rather than multiplying by its reciprocal, stays
/// finite when the magnitude is subnormal.
fn unit_phasor(z: Complex) -> Option<Complex> {
    let magnitude = z.norm();
    (magnitude > 0.0).then(|| Complex::new(z.re / magnitude, z.im / magnitude))
}

/// Searches within one frame either side of the integer peak, clipped to
/// `±max_lag`, for the lag of the largest correlation. The correlation is
/// sinc-shaped about a pure delay, so a parabola through three whole lags
/// misplaces it; instead each step evaluates the correlation at evenly spaced
/// fractional lags and narrows the interval about the best of them. Returns
/// the lag and the unnormalised correlation there.
fn refine(weighted: &[Complex], peak: f32, max_lag: f32) -> (f32, f32) {
    let mut best = (peak, correlation_at(weighted, peak));
    let mut low = (peak - 1.0).max(-max_lag);
    let mut high = (peak + 1.0).min(max_lag);
    for _ in 0..REFINE_STEPS {
        let spacing = (high - low) / (REFINE_POINTS - 1) as f32;
        for i in 0..REFINE_POINTS {
            let lag = low + spacing * i as f32;
            let value = correlation_at(weighted, lag);
            if value > best.1 {
                best = (lag, value);
            }
        }
        low = (best.0 - spacing).max(-max_lag);
        high = (best.0 + spacing).min(max_lag);
    }
    best
}

/// The unnormalised inverse transform of the weighted cross-spectrum at a
/// fractional lag: the band-limited interpolation of the correlation, with
/// each bin taken at its signed frequency so that the result is real for a
/// real pair of signals.
fn correlation_at(weighted: &[Complex], lag: f32) -> f32 {
    let n = weighted.len();
    let mut sum = 0.0f32;
    for (k, w) in weighted.iter().enumerate() {
        let frequency = if k <= n / 2 {
            k as f32
        } else {
            k as f32 - n as f32
        };
        let (sin, cos) = (2.0 * PI * frequency * lag / n as f32).sin_cos();
        sum += w.re * cos - w.im * sin;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixed-seed generator of uniform noise in [-1, 1).
    struct Noise(u64);

    impl Noise {
        fn next(&mut self) -> f32 {
            // xorshift64*
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            let bits = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40;
            bits as f32 / (1u64 << 23) as f32 - 1.0
        }

        fn take(&mut self, count: usize) -> Vec<f32> {
            (0..count).map(|_| self.next()).collect()
        }
    }

    /// Two channels cut from one noise sequence, `b` lagging `a` by `delay`
    /// whole frames.
    fn delayed_noise(len: usize, delay: isize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let margin = delay.unsigned_abs();
        let source = Noise(seed).take(len + margin);
        let start_a = if delay >= 0 { margin } else { 0 };
        let start_b = (start_a as isize - delay) as usize;
        (
            source[start_a..start_a + len].to_vec(),
            source[start_b..start_b + len].to_vec(),
        )
    }

    /// A sum of sinusoids evaluated at `t - delay`, so that a fractional delay
    /// is exact. The components are one per frequency resolution cell of a
    /// `len`-frame window across the whole band, with random phases: the phase
    /// transform weights every bin equally, so a sparse set of tones would
    /// leave most bins to window leakage, whose phase follows the tone's
    /// frequency and not the bin's.
    fn sinusoids(len: usize, delay: f64) -> Vec<f32> {
        let mut noise = Noise(7);
        let count = len / 2;
        let components: Vec<(f64, f64, f64)> = (0..count)
            .map(|m| {
                let frequency = (m as f64 + 0.5) / len as f64;
                let amplitude = 0.5 + (f64::from(noise.next()) + 1.0) / 4.0;
                let phase = std::f64::consts::PI * f64::from(noise.next());
                (frequency, amplitude, phase)
            })
            .collect();
        (0..len)
            .map(|t| {
                let t = t as f64 - delay;
                components
                    .iter()
                    .map(|(f, amp, ph)| amp * (2.0 * std::f64::consts::PI * f * t + ph).sin())
                    .sum::<f64>() as f32
            })
            .collect()
    }

    #[test]
    fn whole_frame_delays_are_recovered() {
        for (delay, seed) in [(0, 1), (1, 2), (3, 3), (-2, 4), (-7, 5), (12, 6)] {
            let (a, b) = delayed_noise(2048, delay, seed);
            let got = gcc_phat(&a, &b, 16).unwrap();
            assert!(
                (got.frames - delay as f32).abs() < 0.05,
                "delay {delay}: got {got:?}"
            );
        }
    }

    #[test]
    fn swapping_the_inputs_negates_the_delay() {
        let (a, b) = delayed_noise(2048, 5, 11);
        let forward = gcc_phat(&a, &b, 16).unwrap();
        let backward = gcc_phat(&b, &a, 16).unwrap();
        assert!(
            (forward.frames + backward.frames).abs() < 1e-3,
            "{forward:?} against {backward:?}"
        );
    }

    #[test]
    fn identical_inputs_give_no_delay_and_a_unit_peak() {
        let a = Noise(21).take(1000);
        let got = gcc_phat(&a, &a, 8).unwrap();
        assert!(got.frames.abs() < 1e-3, "{got:?}");
        assert!((got.peak - 1.0).abs() < 1e-3, "{got:?}");
    }

    #[test]
    fn fractional_delays_are_recovered() {
        for delay in [0.25, 0.5, -0.4, 1.3, -2.75, 4.6] {
            let a = sinusoids(2048, 0.0);
            let b = sinusoids(2048, delay);
            let got = gcc_phat(&a, &b, 8).unwrap();
            assert!(
                (f64::from(got.frames) - delay).abs() < 0.05,
                "delay {delay}: got {got:?}"
            );
        }
    }

    /// `signal` with every sample multiplied by `gain`.
    fn amplified(signal: &[f32], gain: f32) -> Vec<f32> {
        signal.iter().map(|x| x * gain).collect()
    }

    #[test]
    fn delays_are_recovered_at_any_finite_amplitude() {
        // In f32, the square of 1e18 overflows, the square of 1e-30
        // underflows, a sum of samples near f32::MAX overflows, and 1e-42 is
        // subnormal.
        let (a, b) = delayed_noise(1000, 3, 61);
        for (gain_a, gain_b) in [
            (1e18, 1e18),
            (f32::MAX, f32::MAX),
            (1e-30, 1e-30),
            (1e-42, 1e-42),
            (f32::MAX, 1e-42),
        ] {
            let got = gcc_phat(&amplified(&a, gain_a), &amplified(&b, gain_b), 8).unwrap();
            assert!(
                (got.frames - 3.0).abs() < 0.05,
                "gains {gain_a:e} and {gain_b:e}: got {got:?}"
            );
            assert!(
                got.peak.is_finite() && got.peak > 0.5,
                "gains {gain_a:e} and {gain_b:e}: got {got:?}"
            );
        }
    }

    #[test]
    fn identical_inputs_at_the_largest_amplitude_give_a_unit_peak() {
        let a = amplified(&Noise(71).take(1000), f32::MAX);
        let got = gcc_phat(&a, &a, 8).unwrap();
        assert!(got.frames.abs() < 1e-3, "{got:?}");
        assert!((got.peak - 1.0).abs() < 1e-3, "{got:?}");
    }

    #[test]
    fn delays_at_the_bound_of_the_range_are_recovered() {
        for (delay, seed) in [(8, 81), (-8, 82)] {
            let (a, b) = delayed_noise(2048, delay, seed);
            let got = gcc_phat(&a, &b, 8).unwrap();
            assert!(
                (got.frames - delay as f32).abs() < 0.05,
                "delay {delay}: got {got:?}"
            );
        }
        for delay in [7.6, -7.6] {
            let a = sinusoids(2048, 0.0);
            let b = sinusoids(2048, delay);
            let got = gcc_phat(&a, &b, 8).unwrap();
            assert!(
                (f64::from(got.frames) - delay).abs() < 0.05,
                "delay {delay}: got {got:?}"
            );
        }
    }

    /// Out of range, the only promise is that the result stays within it;
    /// where within it is not specified.
    #[test]
    fn a_delay_beyond_the_range_returns_a_lag_within_it() {
        let (a, b) = delayed_noise(2048, 20, 31);
        let got = gcc_phat(&a, &b, 5).unwrap();
        assert!(got.frames.abs() <= 5.0, "{got:?}");
    }

    #[test]
    fn different_lengths_are_refused() {
        assert_eq!(
            gcc_phat(&[0.5; 10], &[0.5; 9], 2),
            Err(DelayError::LengthMismatch { a: 10, b: 9 })
        );
    }

    #[test]
    fn max_lag_not_below_the_length_is_refused() {
        assert_eq!(
            gcc_phat(&[0.5; 4], &[0.5; 4], 4),
            Err(DelayError::MaxLagTooLarge { len: 4, max_lag: 4 })
        );
        assert_eq!(
            gcc_phat(&[], &[], 0),
            Err(DelayError::MaxLagTooLarge { len: 0, max_lag: 0 })
        );
    }

    #[test]
    fn a_non_finite_sample_is_refused() {
        let a = Noise(41).take(64);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut b = a.clone();
            b[10] = bad;
            assert_eq!(gcc_phat(&a, &b, 4), Err(DelayError::NonFinite));
            assert_eq!(gcc_phat(&b, &a, 4), Err(DelayError::NonFinite));
        }
    }

    #[test]
    fn silence_is_refused() {
        let a = Noise(51).take(64);
        let silent = vec![0.0; 64];
        assert_eq!(gcc_phat(&silent, &silent, 4), Err(DelayError::NoSignal));
        assert_eq!(gcc_phat(&a, &silent, 4), Err(DelayError::NoSignal));
    }
}
