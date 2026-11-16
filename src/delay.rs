//! Time-delay estimation between two channels.
//!
//! [`gcc_phat`] is the generalised cross-correlation with the phase transform:
//! the cross-spectrum of the two signals, each bin divided by its own
//! magnitude, transformed back to a correlation over lags. For sound with
//! energy across the whole band, a pure delay between the channels becomes one
//! sharp peak at that lag; [`gcc_phat`] says what happens on other sound.
//!
//! [`reference_delay`] is the same method made fit for finding the echo of a
//! loudspeaker's reference in a microphone, in whole frames: the signals are
//! windowed, only the bins the reference occupies carry weight, and an
//! estimate the signals cannot support is refused.

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
    let spectrum_a = spectrum(a, n, false);
    let spectrum_b = spectrum(b, n, false);

    let (weighted, weighted_bins) = phase_transform(&spectrum_a, &spectrum_b, |_| true);
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

/// The transform of `signal` divided by its largest magnitude, multiplied by
/// the raised-cosine window `0.5 - 0.5 cos(2 pi (i + 0.5) / len)` when
/// `window` is set, and zero-padded to `n` frames. The phase transform
/// ignores each channel's scale, and at unit peak no sum in the transform can
/// overflow and no sample of a quiet channel loses precision to the subnormal
/// range.
fn spectrum(signal: &[f32], n: usize, window: bool) -> Vec<Complex> {
    let mut data = vec![Complex::ZERO; n];
    let peak = signal.iter().fold(0.0f32, |peak, x| peak.max(x.abs()));
    if peak > 0.0 {
        let len = signal.len() as f32;
        for (i, (slot, &x)) in data.iter_mut().zip(signal).enumerate() {
            let taper = if window {
                0.5 - 0.5 * (2.0 * PI * (i as f32 + 0.5) / len).cos()
            } else {
                1.0
            };
            slot.re = x / peak * taper;
        }
    }
    fft::forward(&mut data);
    data
}

/// The share of the band below which [`reference_delay`] refuses a
/// reference as too narrow.
const MIN_OCCUPANCY: f32 = 0.125;

/// A bin's share of the reference's mean power per bin at and above which
/// the bin counts as occupied.
const OCCUPIED_POWER: f32 = 1e-6;

/// The correlation's peak over its floor below which [`reference_delay`]
/// finds no clear delay.
const MIN_PROMINENCE: f32 = 16.0;

/// Where the echo of a reference sits in a microphone signal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReferenceDelay {
    /// Whole frames by which the echo in `mic` lags `reference`. Negative
    /// when it leads, which a live device cannot produce and a reference
    /// placed on a recording afterwards can.
    pub frames: isize,
    /// The correlation's magnitude at that lag over the correlation's floor.
    pub prominence: f32,
}

/// Why [`reference_delay`] refused its input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AlignError {
    /// The two channels have different lengths.
    LengthMismatch { reference: usize, mic: usize },
    /// The channels are empty, or shorter than four times the largest delay
    /// searched.
    MaxDelayTooLarge { len: usize, max_delay: usize },
    /// A sample is NaN or infinite.
    NonFinite,
    /// The reference fills less than an eighth of the band; a silent
    /// reference fills none of it.
    ReferenceTooNarrow { occupancy: f32 },
    /// No lag within the range stands clear of the correlation's floor. The
    /// prominence is 0 when the microphone is silent or the floor is zero.
    NoClearDelay { prominence: f32 },
}

impl fmt::Display for AlignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlignError::LengthMismatch { reference, mic } => {
                write!(
                    f,
                    "channel lengths differ: reference {reference}, microphone {mic} frames"
                )
            }
            AlignError::MaxDelayTooLarge { len, max_delay } => write!(
                f,
                "{len} frames cannot carry a search of {max_delay}: at least one frame and \
                 four times the largest delay are needed"
            ),
            AlignError::NonFinite => write!(f, "a sample is not finite"),
            AlignError::ReferenceTooNarrow { occupancy } => write!(
                f,
                "the reference fills {occupancy} of the band, below the {MIN_OCCUPANCY} required"
            ),
            AlignError::NoClearDelay { prominence } => write!(
                f,
                "no delay stands clear: prominence {prominence}, below the {MIN_PROMINENCE} \
                 required"
            ),
        }
    }
}

impl std::error::Error for AlignError {}

/// Finds where the echo of `reference` sits in `mic`, in whole frames,
/// searched within `-max_delay..=max_delay`.
///
/// Both channels must have the same length, at least one frame and at least
/// four times `max_delay`, and every sample must be finite. The method is
/// the generalised cross-correlation with the phase transform over windowed
/// signals, weighted only in the bins the reference occupies. The result is
/// the lag of the correlation of largest magnitude within the range, the
/// lowest such lag on a tie, with its prominence: that magnitude over the
/// median of the correlation's magnitude across lags within a quarter of the
/// length either side of zero. An echo inverted by the playback or capture
/// chain is found as an upright one is.
///
/// An estimate that is not to be trusted is an error, not a value:
/// a reference occupying less than an eighth of the band, such as a few
/// tones or sound confined to a narrow band, is
/// [`AlignError::ReferenceTooNarrow`]; a prominence below 16, including a
/// silent microphone, is [`AlignError::NoClearDelay`].
///
/// The result is the strongest arrival, which in a room is the direct path
/// and on a device's own loudspeaker dominates. The delay must hold still
/// over the span given. A span in which the reference is silent part of the
/// time is accepted when the rest of it fills the band. With a near-end
/// talker in the span the estimate holds but the prominence falls.
/// Each channel is scaled by its own largest magnitude, so any finite
/// amplitude is accepted.
pub fn reference_delay(
    reference: &[f32],
    mic: &[f32],
    max_delay: usize,
) -> Result<ReferenceDelay, AlignError> {
    if reference.len() != mic.len() {
        return Err(AlignError::LengthMismatch {
            reference: reference.len(),
            mic: mic.len(),
        });
    }
    let len = reference.len();
    if len == 0 || len / 4 < max_delay {
        return Err(AlignError::MaxDelayTooLarge { len, max_delay });
    }
    if !reference.iter().chain(mic).all(|x| x.is_finite()) {
        return Err(AlignError::NonFinite);
    }

    let n = (len + max_delay + 1).next_power_of_two();
    let spectrum_reference = spectrum(reference, n, true);
    let spectrum_mic = spectrum(mic, n, true);

    // At unit peak the squared magnitudes cannot overflow.
    let powers: Vec<f32> = spectrum_reference
        .iter()
        .map(|z| z.re * z.re + z.im * z.im)
        .collect();
    let mean_power = powers.iter().map(|&p| f64::from(p)).sum::<f64>() / n as f64;
    let threshold = OCCUPIED_POWER * mean_power as f32;
    let occupied: Vec<bool> = powers.iter().map(|&p| p > 0.0 && p >= threshold).collect();
    let occupancy = occupied.iter().filter(|&&o| o).count() as f32 / n as f32;
    if occupancy < MIN_OCCUPANCY {
        return Err(AlignError::ReferenceTooNarrow { occupancy });
    }

    let (mut correlation, weighted_bins) =
        phase_transform(&spectrum_reference, &spectrum_mic, |k| occupied[k]);
    if weighted_bins == 0 {
        return Err(AlignError::NoClearDelay { prominence: 0.0 });
    }
    fft::inverse(&mut correlation);
    let scale = n as f32 / weighted_bins as f32;
    let lag_value = |lag: isize| correlation[lag.rem_euclid(n as isize) as usize].re * scale;

    // The echo's polarity is the playback and capture chain's, so the peak
    // is searched by magnitude.
    let bound = max_delay as isize;
    let mut best = -bound;
    for lag in -bound..=bound {
        if lag_value(lag).abs() > lag_value(best).abs() {
            best = lag;
        }
    }

    let quarter = (len / 4) as isize;
    let mut magnitudes: Vec<f32> = (-quarter..=quarter)
        .map(|lag| lag_value(lag).abs())
        .collect();
    let middle = magnitudes.len() / 2;
    let floor = *magnitudes.select_nth_unstable_by(middle, f32::total_cmp).1;
    if floor <= 0.0 {
        return Err(AlignError::NoClearDelay { prominence: 0.0 });
    }
    let prominence = lag_value(best).abs() / floor;
    if prominence < MIN_PROMINENCE {
        return Err(AlignError::NoClearDelay { prominence });
    }
    Ok(ReferenceDelay {
        frames: best,
        prominence,
    })
}

/// The phase transform of two spectra of one length: each bin of conj(A) B
/// divided by its magnitude, formed as the product of the two bins' unit
/// phasors so that no product of magnitudes can overflow or underflow, and
/// the number of bins so weighted. A bin `include` rejects, and a bin of zero
/// magnitude in either spectrum, which carries no phase, gets zero.
fn phase_transform(
    a: &[Complex],
    b: &[Complex],
    include: impl Fn(usize) -> bool,
) -> (Vec<Complex>, usize) {
    let mut weighted = vec![Complex::ZERO; a.len()];
    let mut weighted_bins = 0usize;
    for (k, (slot, (x, y))) in weighted.iter_mut().zip(a.iter().zip(b)).enumerate() {
        if !include(k) {
            continue;
        }
        if let (Some(x), Some(y)) = (unit_phasor(*x), unit_phasor(*y)) {
            *slot = x.conj().mul(y);
            weighted_bins += 1;
        }
    }
    (weighted, weighted_bins)
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
    use crate::test_signals::{
        Noise, amplified, convolve, low_passed, mean_square, noise_below, sum, test_path, white,
    };

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

    /// Frames of the test path's echo that precede a cut, so that every cut
    /// holds the echo in its steady state.
    const PATH_LEN: usize = 1200;

    /// A source and its echo through the test path, from which pairs of
    /// `len` frames are cut with the echo up to `max_delay` frames either side
    /// of the reference.
    struct Echo {
        source: Vec<f32>,
        echo: Vec<f32>,
        margin: usize,
        len: usize,
    }

    impl Echo {
        fn new(source: Vec<f32>, len: usize, max_delay: usize) -> Echo {
            let margin = PATH_LEN + max_delay;
            assert_eq!(source.len(), len + 2 * margin, "source of the wrong length");
            let echo = convolve(&source, &test_path(200.0, 3));
            Echo {
                source,
                echo,
                margin,
                len,
            }
        }

        /// The reference and the microphone, the echo in the microphone
        /// lagging the reference by `delay`, with white noise `below_db` under
        /// the echo's power added to the microphone.
        fn pair(&self, delay: isize, below_db: f64, seed: u64) -> (Vec<f32>, Vec<f32>) {
            let reference = self.source[self.margin..self.margin + self.len].to_vec();
            let start = (self.margin as isize - delay) as usize;
            let echo = &self.echo[start..start + self.len];
            let mic = sum(echo, &noise_below(echo, below_db, seed));
            (reference, mic)
        }
    }

    /// The delays checked at each length and range.
    const SHORT_DELAYS: [isize; 8] = [-2000, -37, -1, 0, 1, 37, 800, 2000];

    fn sources(len: usize, max_delay: usize) -> [(&'static str, Vec<f32>); 2] {
        let total = len + 2 * (PATH_LEN + max_delay);
        [
            ("white", white(total, 0.01, 101)),
            ("low-passed", low_passed(total, 0.01, 102)),
        ]
    }

    #[test]
    fn reference_delays_are_recovered_either_side_of_zero_and_of_either_polarity() {
        for (len, max_delay, delays) in [
            (8000, 2000, &SHORT_DELAYS[..]),
            (32000, 8000, &[-8000, 8000][..]),
        ] {
            for (kind, source) in sources(len, max_delay) {
                let echo = Echo::new(source, len, max_delay);
                for (i, &delay) in delays.iter().enumerate() {
                    let (reference, mic) = echo.pair(delay, 15.0, 200 + i as u64);
                    let got = reference_delay(&reference, &mic, max_delay).unwrap();
                    assert_eq!(got.frames, delay, "{kind}, {len} frames: {got:?}");
                    assert!(got.prominence >= 16.0, "{kind}: {got:?}");
                    let inverted = reference_delay(&reference, &amplified(&mic, -1.0), max_delay);
                    assert_eq!(inverted, Ok(got), "{kind}, {len} frames, inverted");
                }
            }
        }
    }

    #[test]
    fn the_floor_does_not_depend_on_the_range() {
        let echo = Echo::new(white(8000 + 2 * (PATH_LEN + 64), 0.01, 111), 8000, 64);
        for (max_delay, delay) in [(8, -8), (8, 0), (8, 5), (64, -40), (64, 64)] {
            let (reference, mic) = echo.pair(delay, 15.0, 112);
            let got = reference_delay(&reference, &mic, max_delay).unwrap();
            assert_eq!(got.frames, delay, "range {max_delay}: {got:?}");
        }
    }

    #[test]
    fn a_reference_silent_for_its_first_half_is_accepted() {
        let len = 8000;
        let max_delay = 2000;
        let mut source = white(len + 2 * (PATH_LEN + max_delay), 0.01, 121);
        let silent_until = PATH_LEN + max_delay + len / 2;
        source[..silent_until].fill(0.0);
        let echo = Echo::new(source, len, max_delay);
        let (reference, mic) = echo.pair(37, 15.0, 122);
        let got = reference_delay(&reference, &mic, max_delay).unwrap();
        assert_eq!(got.frames, 37, "{got:?}");
    }

    #[test]
    fn unrelated_signals_have_no_clear_delay() {
        for (len, max_delay) in [(8000, 2000), (32000, 8000)] {
            for seed in [131, 133, 135] {
                for (kind, reference, mic) in [
                    ("white", white(len, 0.01, seed), white(len, 0.01, seed + 1)),
                    (
                        "low-passed",
                        low_passed(len, 0.01, seed),
                        low_passed(len, 0.01, seed + 1),
                    ),
                ] {
                    let got = reference_delay(&reference, &mic, max_delay);
                    assert!(
                        matches!(got, Err(AlignError::NoClearDelay { .. })),
                        "{kind}, {len} frames, seed {seed}: {got:?}"
                    );
                }
            }
        }
    }

    /// The sum of sinusoids at every frequency resolution cell of a
    /// `len`-frame span between `low` and `high` cycles per frame, at random
    /// phases, evaluated at `t - delay`.
    fn band(len: usize, low: f64, high: f64, delay: usize, seed: u64) -> Vec<f32> {
        let mut noise = Noise(seed);
        let components: Vec<(f64, f64)> = (0..len)
            .map(|m| (m as f64 + 0.5) / len as f64)
            .filter(|f| (low..=high).contains(f))
            .map(|f| (f, std::f64::consts::PI * f64::from(noise.next())))
            .collect();
        (0..len)
            .map(|t| {
                let t = t as f64 - delay as f64;
                components
                    .iter()
                    .map(|(f, phase)| (2.0 * std::f64::consts::PI * f * t + phase).sin())
                    .sum::<f64>() as f32
            })
            .collect()
    }

    fn assert_too_narrow(reference: &[f32], mic: &[f32], case: &str) {
        let got = reference_delay(reference, mic, 2000);
        assert!(
            matches!(got, Err(AlignError::ReferenceTooNarrow { occupancy }) if occupancy < 0.05),
            "{case}: {got:?}"
        );
    }

    #[test]
    fn a_narrow_band_reference_is_refused() {
        let len = 8000;
        let tone = |delay: usize| -> Vec<f32> {
            (0..len)
                .map(|t| {
                    let t = t as f64 - delay as f64;
                    (2.0 * std::f64::consts::PI * 0.0625 * t).sin() as f32
                })
                .collect()
        };
        let reference = tone(0);
        let echo = tone(37);
        assert_too_narrow(&reference, &echo, "tone");
        let noise = white(len, mean_square(&echo) / 100.0, 141);
        assert_too_narrow(&reference, &sum(&echo, &noise), "tone, noisy microphone");

        let reference = band(len, 900.0 / 16000.0, 1100.0 / 16000.0, 0, 142);
        let echo = band(len, 900.0 / 16000.0, 1100.0 / 16000.0, 37, 142);
        assert_too_narrow(&reference, &echo, "900 to 1100 Hz");
    }

    #[test]
    fn a_reference_filling_the_speech_band_is_accepted() {
        let reference = band(8000, 200.0 / 16000.0, 3400.0 / 16000.0, 0, 151);
        let mic = band(8000, 200.0 / 16000.0, 3400.0 / 16000.0, 37, 151);
        let got = reference_delay(&reference, &mic, 2000).unwrap();
        assert_eq!(got.frames, 37, "{got:?}");
    }

    #[test]
    fn a_silent_channel_is_refused() {
        let signal = white(8000, 0.01, 161);
        let silent = vec![0.0; 8000];
        assert_eq!(
            reference_delay(&silent, &signal, 2000),
            Err(AlignError::ReferenceTooNarrow { occupancy: 0.0 })
        );
        assert_eq!(
            reference_delay(&signal, &silent, 2000),
            Err(AlignError::NoClearDelay { prominence: 0.0 })
        );
    }

    #[test]
    fn reference_delay_refuses_invalid_input() {
        let signal = white(64, 0.01, 171);
        assert_eq!(
            reference_delay(&signal, &signal[..63], 4),
            Err(AlignError::LengthMismatch {
                reference: 64,
                mic: 63
            })
        );
        assert_eq!(
            reference_delay(&signal[..63], &signal[..63], 16),
            Err(AlignError::MaxDelayTooLarge {
                len: 63,
                max_delay: 16
            })
        );
        assert!(reference_delay(&signal, &signal, 16).is_ok());
        assert_eq!(
            reference_delay(&[], &[], 0),
            Err(AlignError::MaxDelayTooLarge {
                len: 0,
                max_delay: 0
            })
        );
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut broken = signal.clone();
            broken[10] = bad;
            assert_eq!(
                reference_delay(&signal, &broken, 4),
                Err(AlignError::NonFinite)
            );
            assert_eq!(
                reference_delay(&broken, &signal, 4),
                Err(AlignError::NonFinite)
            );
        }
    }

    #[test]
    fn reference_delay_holds_at_any_finite_amplitude() {
        let echo = Echo::new(white(8000 + 2 * (PATH_LEN + 2000), 0.01, 181), 8000, 2000);
        let (reference, mic) = echo.pair(37, 15.0, 182);
        for gain in [1e-30, 1e18] {
            let got = reference_delay(&amplified(&reference, gain), &amplified(&mic, gain), 2000)
                .unwrap();
            assert_eq!(got.frames, 37, "gain {gain:e}: {got:?}");
        }
    }
}
