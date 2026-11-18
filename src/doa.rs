//! The direction of arrival of the dominant source on a line of microphones.
//!
//! [`Direction`] estimates it at a fixed interval from the microphones' last
//! stretch of signal. Angles and channels follow the conventions of
//! [`crate::array`].
//!
//! "Frame" is the crate's unit of time, one sample instant, here as in
//! [`crate::filter`] and [`crate::echo`]. The analysis length is a
//! [`DirectionConfig::window`] of frames, and a reading's place in the stream
//! is [`Reading::frame`], a count of frames.

use crate::array::{BlockFault, LineArray, check_block};
use crate::delay::{spectrum, unit_phasor};
use crate::fft::Complex;
use std::f64::consts::PI;
use std::fmt;

/// The shortest and longest analysis windows accepted, in frames. The cap
/// bounds the history a [`Direction`] keeps: 65 536 frames is 4 s at 16 kHz.
const MIN_WINDOW: usize = 8;
const MAX_WINDOW: usize = 1 << 16;

/// How [`Direction`] reads its microphones. There is no default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectionConfig {
    /// The analysis length in frames: a power of two from 8 to 65 536. The
    /// tested value is 1024.
    pub window: usize,
    /// The interval between readings in frames, from 1 to `window`. The
    /// tested value is 256.
    pub hop: usize,
    /// The least peak that counts as an estimate, within `(0, 1)`. The
    /// tested value is 0.2.
    pub min_peak: f32,
}

/// One estimate of the direction of the dominant source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reading {
    /// How many frames had been fed when the reading was made. The reading
    /// describes the last `window` of them.
    pub frame: u64,
    /// The angle from the array's axis in radians, within `[0, pi]`, or
    /// `None` for no estimate.
    pub angle: Option<f32>,
    /// The steered response at its largest: 1 for one plane wave alone,
    /// about 0 for sound independent from microphone to microphone, 0 when
    /// no bin was used. Reported with or without an angle.
    pub peak: f32,
}

/// Why [`Direction`] refused its configuration or its input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectionError {
    /// The window is not a power of two from 8 to 65 536.
    WindowOutOfRange { window: usize },
    /// The hop is not within `1..=window`.
    HopOutOfRange { hop: usize },
    /// The least peak is not within `(0, 1)`.
    MinPeakOutOfRange,
    /// The block has `got` microphone channels where the array has
    /// `expected`.
    MicCount { expected: usize, got: usize },
    /// The block's channels differ in length.
    LengthMismatch,
    /// A sample is NaN or infinite.
    NonFinite,
}

impl fmt::Display for DirectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DirectionError::WindowOutOfRange { window } => write!(
                f,
                "window {window} is not a power of two from {MIN_WINDOW} to {MAX_WINDOW}"
            ),
            DirectionError::HopOutOfRange { hop } => {
                write!(f, "hop {hop} is not within 1..=window")
            }
            DirectionError::MinPeakOutOfRange => {
                write!(f, "the least peak is not within (0, 1)")
            }
            DirectionError::MicCount { expected, got } => write!(
                f,
                "{got} microphone channels where the array has {expected}"
            ),
            DirectionError::LengthMismatch => write!(f, "the block's channels differ in length"),
            DirectionError::NonFinite => write!(f, "a sample is not finite"),
        }
    }
}

impl From<BlockFault> for DirectionError {
    fn from(fault: BlockFault) -> DirectionError {
        match fault {
            BlockFault::MicCount { expected, got } => DirectionError::MicCount { expected, got },
            BlockFault::LengthMismatch => DirectionError::LengthMismatch,
            BlockFault::NonFinite => DirectionError::NonFinite,
        }
    }
}

impl std::error::Error for DirectionError {}

/// The direction of the dominant source, read every `hop` frames from the
/// last `window`, fed in blocks.
///
/// **The method** is the steered response power with the phase transform.
/// Each microphone's last `window` frames are tapered by the raised cosine
/// `0.5 - 0.5 cos(2 pi (i + 0.5) / window)` and transformed. Of the bins
/// strictly between zero and half the rate, a bin is used when every
/// microphone's bin has a non-zero magnitude; for each used bin and each
/// pair of microphones the cross-spectrum is divided by its magnitude. For
/// each of `G = 2 * ceil(4 (M - 1) * spacing) + 1` candidate directions,
/// evenly spaced in `cos(angle)` from -1 to 1 and so a quarter frame or less
/// apart in delay across the whole array, the pairs' terms are turned back
/// by the delay that direction implies and summed. The sum is divided by the
/// pairs times the bins used, so one plane wave alone gives 1 and sound
/// independent from microphone to microphone about 0. The estimate is the
/// candidate with the largest sum, moved between candidates by a parabola
/// through its two neighbours when it has two and both lie below it,
/// clamped to `[-1, 1]`; the angle is its arccosine.
///
/// **No estimate** when fewer than half the bins are used, which covers
/// silence, or when the peak is below `min_peak`, which covers sound that is
/// not coherent across the microphones.
///
/// **What it is not.** It has no notion of speech: it reports the dominant
/// coherent source, whatever that is. Sound that is the same on every
/// microphone, such as hum or a common electrical tone, is a coherent source
/// at broadside and is reported as one; [`crate::filter::HighPass`] on every
/// channel removes rumble and leaves the delays between channels as they
/// were. Every bin weighs alike, so a source that fills little of the band
/// has a low peak. Near the axis `cos(angle)` barely changes with the angle,
/// so there the angle is coarse and can snap to the axis itself. Two sources
/// of nearly equal strength can take the estimate in turn from reading to
/// reading.
///
/// **Delays.** The first reading comes when `window` frames have been fed
/// and one every `hop` frames after. A reading describes the last `window`
/// frames fed.
///
/// **Cost, per reading**: `M` transforms of `window` points and
/// `(M - 1) * G * window / 2` complex multiply-adds for the search, each
/// candidate's phasor advanced from bin to bin. For four microphones 1.54
/// frames apart at a window of 1024 that is four transforms and about 60 000
/// multiply-adds. Memory is `M * window` samples and `(M - 1) * G` phasors.
#[derive(Clone, Debug)]
pub struct Direction {
    mics: usize,
    config: DirectionConfig,
    /// The candidates' cosines, from -1 to 1.
    candidates: Vec<f64>,
    /// For microphones `d` apart and candidate `g`, at `(d - 1) * G + g`, the
    /// phasor that turns a bin's term back by that candidate's delay from
    /// one bin to the next: `e^(-2 pi i d spacing cos / window)`.
    steps: Vec<(f64, f64)>,
    /// Per microphone, its last `window` frames, oldest first, zero before
    /// the start; during a block the block follows them.
    lines: Vec<Vec<f32>>,
    /// How many frames have been fed.
    fed: u64,
    /// The count of frames fed at which the next reading falls due.
    due: u64,
}

impl Direction {
    /// A direction estimate for `array`, nothing fed. Refuses a `window`,
    /// `hop` or `min_peak` out of range.
    pub fn new(array: LineArray, config: DirectionConfig) -> Result<Direction, DirectionError> {
        let window = config.window;
        if !(window.is_power_of_two() && (MIN_WINDOW..=MAX_WINDOW).contains(&window)) {
            return Err(DirectionError::WindowOutOfRange { window });
        }
        if !(1..=window).contains(&config.hop) {
            return Err(DirectionError::HopOutOfRange { hop: config.hop });
        }
        if !(config.min_peak > 0.0 && config.min_peak < 1.0) {
            return Err(DirectionError::MinPeakOutOfRange);
        }
        let mics = array.mics();
        let spacing = f64::from(array.spacing());
        let half = (4.0 * (mics - 1) as f64 * spacing).ceil() as usize;
        let count = 2 * half + 1;
        let candidates: Vec<f64> = (0..count)
            .map(|g| (g as f64 - half as f64) / half as f64)
            .collect();
        let steps = (1..mics)
            .flat_map(|d| {
                candidates.iter().map(move |cos| {
                    let turn = -2.0 * PI * d as f64 * spacing * cos / window as f64;
                    (turn.cos(), turn.sin())
                })
            })
            .collect();
        Ok(Direction {
            mics,
            config,
            candidates,
            steps,
            lines: vec![vec![0.0; window]; mics],
            fed: 0,
            due: window as u64,
        })
    }

    /// Appends to `readings` every reading that falls due in the block,
    /// continuing from the previous call, so that any cut of a signal into
    /// blocks gives the same readings.
    ///
    /// `mics` holds one slice per microphone, every slice of one length. A
    /// block with the wrong count of channels, channels of different lengths
    /// or a sample that is not finite is refused whole, leaving the estimate
    /// and `readings` untouched.
    pub fn process(
        &mut self,
        mics: &[&[f32]],
        readings: &mut Vec<Reading>,
    ) -> Result<(), DirectionError> {
        check_block(mics, self.mics)?;
        self.feed(mics, readings);
        Ok(())
    }

    /// [`Direction::process`] for a block `check_block` has accepted for
    /// this estimate's count of microphones.
    pub(crate) fn feed(&mut self, mics: &[&[f32]], readings: &mut Vec<Reading>) {
        let len = mics[0].len();
        for (line, block) in self.lines.iter_mut().zip(mics) {
            line.extend_from_slice(block);
        }
        let start = self.fed;
        let end = start + len as u64;
        let window = self.config.window;
        while self.due <= end {
            let offset = (self.due - start) as usize;
            let spans: Vec<&[f32]> = self
                .lines
                .iter()
                .map(|line| &line[offset..offset + window])
                .collect();
            let (angle, peak) = self.estimate(&spans);
            readings.push(Reading {
                frame: self.due,
                angle,
                peak,
            });
            self.due += self.config.hop as u64;
        }
        for line in &mut self.lines {
            line.drain(..len);
        }
        self.fed = end;
    }

    /// The angle, or `None`, and the peak for one window per microphone.
    fn estimate(&self, spans: &[&[f32]]) -> (Option<f32>, f32) {
        let window = self.config.window;
        let bins = window / 2;
        let spectra: Vec<Vec<Complex>> = spans.iter().map(|s| spectrum(s, window, true)).collect();
        // For microphones `d` apart, at `(d - 1) * bins + k`: the sum over
        // every such pair of bin `k`'s cross-spectrum divided by its
        // magnitude, zero for a bin not used.
        let mut cross = vec![Complex::ZERO; (self.mics - 1) * bins];
        let mut phasors = vec![Complex::ZERO; self.mics];
        let mut used = 0usize;
        for k in 1..bins {
            let whole = spectra
                .iter()
                .zip(&mut phasors)
                .all(|(s, slot)| unit_phasor(s[k]).map(|u| *slot = u).is_some());
            if !whole {
                continue;
            }
            used += 1;
            for d in 1..self.mics {
                let term = &mut cross[(d - 1) * bins + k];
                for i in 0..self.mics - d {
                    *term = term.add(phasors[i].conj().mul(phasors[i + d]));
                }
            }
        }
        if used == 0 {
            return (None, 0.0);
        }
        let pairs = self.mics * (self.mics - 1) / 2;
        let scale = 1.0 / (pairs * used) as f64;
        let count = self.candidates.len();
        let sums: Vec<f64> = (0..count)
            .map(|g| {
                let mut total = 0.0;
                for d in 1..self.mics {
                    let step = self.steps[(d - 1) * count + g];
                    let mut turn = step;
                    for term in &cross[(d - 1) * bins + 1..d * bins] {
                        total += f64::from(term.re) * turn.0 - f64::from(term.im) * turn.1;
                        turn = (
                            turn.0 * step.0 - turn.1 * step.1,
                            turn.0 * step.1 + turn.1 * step.0,
                        );
                    }
                }
                total * scale
            })
            .collect();
        let best = (1..count).fold(0, |best, g| if sums[g] > sums[best] { g } else { best });
        let peak = sums[best] as f32;
        if 2 * used < bins - 1 || peak < self.config.min_peak {
            return (None, peak);
        }
        let mut cos = self.candidates[best];
        if best > 0 && best + 1 < count {
            let (below, here, above) = (sums[best - 1], sums[best], sums[best + 1]);
            if below < here && above < here {
                let offset = 0.5 * (below - above) / (below - 2.0 * here + above);
                cos += offset * (self.candidates[1] - self.candidates[0]);
            }
        }
        (Some(cos.clamp(-1.0, 1.0).acos() as f32), peak)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_signals::{
        DEGREE, low_passed, plane_wave, sum, test_array, tilted_plane_wave, white, with_noise,
    };

    const CONFIG: DirectionConfig = DirectionConfig {
        window: 1024,
        hop: 256,
        min_peak: 0.2,
    };

    /// Every reading of `mics` fed as one block to a fresh estimate.
    fn readings(mics: &[Vec<f32>]) -> Vec<Reading> {
        let mut direction = Direction::new(test_array(), CONFIG).unwrap();
        let inputs: Vec<&[f32]> = mics.iter().map(Vec::as_slice).collect();
        let mut out = Vec::new();
        direction.process(&inputs, &mut out).unwrap();
        out
    }

    /// The largest error of `readings` from `angle`, in degrees. Panics on a
    /// reading without an angle.
    fn worst_error(readings: &[Reading], angle: f32) -> f32 {
        readings
            .iter()
            .map(|r| {
                let got = r.angle.unwrap_or_else(|| panic!("no angle in {r:?}"));
                (got - angle).abs() / DEGREE
            })
            .fold(0.0, f32::max)
    }

    #[test]
    fn the_candidates_are_a_quarter_frame_apart_or_less() {
        let direction = Direction::new(test_array(), CONFIG).unwrap();
        assert_eq!(direction.candidates.len(), 39);
        assert_eq!(direction.candidates[0], -1.0);
        assert_eq!(direction.candidates[19], 0.0);
        assert_eq!(direction.candidates[38], 1.0);
        assert_eq!(direction.steps.len(), 3 * 39);
    }

    #[test]
    fn one_plane_wave_alone_peaks_near_one() {
        for angle in [30.0, 90.0, 150.0] {
            let wave = plane_wave(&test_array(), angle * DEGREE, (0.0, 0.5), 4096, 0.1, 501);
            for reading in readings(&wave.mics) {
                assert!(reading.peak >= 0.95, "source at {angle}: {reading:?}");
                assert!(reading.peak <= 1.0 + 1e-5, "source at {angle}: {reading:?}");
            }
        }
    }

    #[test]
    fn a_white_source_over_noise_is_found_within_three_degrees_off_the_axis() {
        for step in 2..=10 {
            let angle = 15.0 * step as f32 * DEGREE;
            let wave = plane_wave(&test_array(), angle, (0.0, 0.5), 8192, 0.1, 511 + step);
            let found = readings(&with_noise(&wave.mics, 0.01, 521 + 10 * step));
            assert_eq!(found.len(), 29);
            let error = worst_error(&found, angle);
            assert!(error <= 3.0, "source at {}: {error:.2} degrees", 15 * step);
        }
    }

    #[test]
    fn a_white_source_over_noise_near_the_axis_is_found_within_twenty_degrees() {
        for degrees in [0.0, 15.0, 165.0, 180.0] {
            let angle = degrees * DEGREE;
            let wave = plane_wave(&test_array(), angle, (0.0, 0.5), 8192, 0.1, 531);
            let found = readings(&with_noise(&wave.mics, 0.01, 541));
            let error = worst_error(&found, angle);
            assert!(error <= 20.0, "source at {degrees}: {error:.2} degrees");
        }
    }

    #[test]
    fn a_speech_like_source_over_noise_is_found_within_eight_degrees() {
        for step in 2..=10 {
            let angle = 15.0 * step as f32 * DEGREE;
            let wave = tilted_plane_wave(&test_array(), angle, 8192, 0.1, 551 + step);
            let found = readings(&with_noise(&wave.mics, 0.01, 561 + 10 * step));
            let error = worst_error(&found, angle);
            assert!(error <= 8.0, "source at {}: {error:.2} degrees", 15 * step);
        }
    }

    #[test]
    fn silence_gives_no_estimate_and_no_peak() {
        let found = readings(&vec![vec![0.0; 4096]; 4]);
        assert_eq!(found.len(), 13);
        for reading in found {
            assert_eq!(reading.angle, None);
            assert_eq!(reading.peak, 0.0);
        }
    }

    #[test]
    fn independent_noise_gives_no_estimate() {
        let len = 16384;
        let whites: Vec<Vec<f32>> = (0..4).map(|m| white(len, 0.1, 571 + m)).collect();
        let lows: Vec<Vec<f32>> = (0..4).map(|m| low_passed(len, 0.1, 581 + m)).collect();
        for (name, mics) in [("white", whites), ("low-passed", lows)] {
            for reading in readings(&mics) {
                assert_eq!(reading.angle, None, "{name}: {reading:?}");
                assert!(reading.peak < 0.2, "{name}: {reading:?}");
            }
        }
    }

    #[test]
    fn a_weaker_second_source_does_not_move_the_estimate() {
        for (strong, weak) in [(60.0, 120.0), (90.0, 30.0), (135.0, 45.0)] {
            let len = 8192;
            let a = plane_wave(&test_array(), strong * DEGREE, (0.0, 0.5), len, 0.1, 591);
            let b = plane_wave(&test_array(), weak * DEGREE, (0.0, 0.5), len, 0.025, 601);
            let both: Vec<Vec<f32>> = a.mics.iter().zip(&b.mics).map(|(x, y)| sum(x, y)).collect();
            let found = readings(&with_noise(&both, 0.01, 611));
            let error = worst_error(&found, strong * DEGREE);
            assert!(error <= 5.0, "{strong} over {weak}: {error:.2} degrees");
        }
    }

    #[test]
    fn readings_fall_due_after_a_window_and_every_hop_after() {
        let len = 3000;
        let mics: Vec<Vec<f32>> = (0..4).map(|m| white(len, 0.1, 621 + m)).collect();
        for (window, hop) in [(1024, 256), (8, 1), (64, 64), (256, 100)] {
            let config = DirectionConfig {
                window,
                hop,
                min_peak: 0.2,
            };
            let mut direction = Direction::new(test_array(), config).unwrap();
            let mut found = Vec::new();
            for start in (0..len).step_by(333) {
                let end = (start + 333).min(len);
                let inputs: Vec<&[f32]> = mics.iter().map(|c| &c[start..end]).collect();
                direction.process(&inputs, &mut found).unwrap();
            }
            let frames: Vec<u64> = found.iter().map(|r| r.frame).collect();
            let want: Vec<u64> = (window..=len).step_by(hop).map(|f| f as u64).collect();
            assert_eq!(frames, want, "window {window}, hop {hop}");
        }
    }

    #[test]
    fn a_reading_describes_the_last_window_fed() {
        let len = 4096;
        let wave = plane_wave(&test_array(), 70.0 * DEGREE, (0.0, 0.5), len, 0.1, 631);
        let mics = with_noise(&wave.mics, 0.01, 641);
        let whole = readings(&mics);
        for reading in &whole {
            let end = reading.frame as usize;
            let tail: Vec<Vec<f32>> = mics.iter().map(|c| c[end - 1024..end].to_vec()).collect();
            let alone = readings(&tail);
            assert_eq!(alone.len(), 1);
            assert_eq!(
                alone[0].angle.map(f32::to_bits),
                reading.angle.map(f32::to_bits)
            );
            assert_eq!(alone[0].peak.to_bits(), reading.peak.to_bits());
        }
    }

    #[test]
    fn any_cut_into_blocks_gives_the_same_readings() {
        let len = 4096;
        let wave = plane_wave(&test_array(), 50.0 * DEGREE, (0.0, 0.5), len, 0.1, 651);
        let mics = with_noise(&wave.mics, 0.01, 661);
        let whole = readings(&mics);
        let key = |r: &Reading| (r.frame, r.angle.map(f32::to_bits), r.peak.to_bits());
        for block in [1, 7, 160] {
            let mut direction = Direction::new(test_array(), CONFIG).unwrap();
            let mut found = Vec::new();
            for start in (0..len).step_by(block) {
                let end = (start + block).min(len);
                let inputs: Vec<&[f32]> = mics.iter().map(|c| &c[start..end]).collect();
                direction.process(&inputs, &mut found).unwrap();
            }
            let got: Vec<_> = found.iter().map(key).collect();
            let want: Vec<_> = whole.iter().map(key).collect();
            assert_eq!(got, want, "blocks of {block}");
        }
    }

    #[test]
    fn a_bad_configuration_is_refused() {
        let with = |window, hop, min_peak| {
            Direction::new(
                test_array(),
                DirectionConfig {
                    window,
                    hop,
                    min_peak,
                },
            )
            .err()
        };
        for window in [
            0,
            1,
            4,
            6,
            1000,
            1025,
            2 * MAX_WINDOW,
            1 << (usize::BITS - 1),
        ] {
            assert_eq!(
                with(window, 1, 0.2),
                Some(DirectionError::WindowOutOfRange { window }),
                "window {window}"
            );
        }
        for hop in [0, 1025] {
            assert_eq!(
                with(1024, hop, 0.2),
                Some(DirectionError::HopOutOfRange { hop }),
                "hop {hop}"
            );
        }
        for min_peak in [0.0, -0.1, 1.0, 1.5, f32::NAN, f32::INFINITY] {
            assert_eq!(
                with(1024, 256, min_peak),
                Some(DirectionError::MinPeakOutOfRange),
                "min_peak {min_peak}"
            );
        }
        assert_eq!(with(8, 8, 1e-6), None);
        assert_eq!(with(1024, 1024, 0.999), None);
    }

    #[test]
    fn a_bad_block_is_refused_and_changes_nothing() {
        let first: Vec<Vec<f32>> = (0..4).map(|m| white(1000, 0.1, 671 + m)).collect();
        let second: Vec<Vec<f32>> = (0..4).map(|m| white(1000, 0.1, 681 + m)).collect();
        let feed = |direction: &mut Direction, mics: &[Vec<f32>]| -> Vec<Reading> {
            let inputs: Vec<&[f32]> = mics.iter().map(Vec::as_slice).collect();
            let mut found = Vec::new();
            direction.process(&inputs, &mut found).unwrap();
            found
        };
        let mut clean = Direction::new(test_array(), CONFIG).unwrap();
        let mut refused = Direction::new(test_array(), CONFIG).unwrap();
        feed(&mut clean, &first);
        feed(&mut refused, &first);

        let good: Vec<&[f32]> = second.iter().map(Vec::as_slice).collect();
        let short = &second[3][..999];
        let mut nan = second[2].clone();
        nan[500] = f32::NAN;
        let mut infinite = second[1].clone();
        infinite[0] = f32::INFINITY;
        let cases: Vec<(Vec<&[f32]>, DirectionError)> = vec![
            (
                good[..3].to_vec(),
                DirectionError::MicCount {
                    expected: 4,
                    got: 3,
                },
            ),
            (
                vec![good[0], good[1], good[2], good[3], good[0]],
                DirectionError::MicCount {
                    expected: 4,
                    got: 5,
                },
            ),
            (
                vec![good[0], good[1], good[2], short],
                DirectionError::LengthMismatch,
            ),
            (
                vec![good[0], good[1], &nan, good[3]],
                DirectionError::NonFinite,
            ),
            (
                vec![good[0], &infinite, good[2], good[3]],
                DirectionError::NonFinite,
            ),
        ];
        let sentinel = Reading {
            frame: 7,
            angle: Some(1.0),
            peak: 0.5,
        };
        for (inputs, error) in cases {
            let mut found = vec![sentinel];
            assert_eq!(refused.process(&inputs, &mut found), Err(error));
            assert_eq!(
                found,
                vec![sentinel],
                "{error:?}: the readings were touched"
            );
        }
        let want = feed(&mut clean, &second);
        let got = feed(&mut refused, &second);
        assert!(!want.is_empty());
        let key = |r: &Reading| (r.frame, r.angle.map(f32::to_bits), r.peak.to_bits());
        assert_eq!(
            got.iter().map(key).collect::<Vec<_>>(),
            want.iter().map(key).collect::<Vec<_>>()
        );
    }
}
