//! Fixed beams over a line of microphones, and one beam chosen among them.
//!
//! [`FixedBeams`] steers delay-and-sum beams at fixed angles.
//! [`SelectedBeam`] puts out one of them, moving to the beam nearest the
//! direction of the dominant source. Angles and channels follow the
//! conventions of [`crate::array`].

use crate::array::{BlockFault, LineArray, check_block};
use crate::doa::{Direction, DirectionConfig, DirectionError};
use std::f64::consts::PI;
use std::fmt;

/// Half the length of a fractional-delay filter, in frames: the windowed
/// sinc is zero from this far either side of its delay.
const HALF_WIDTH: usize = 16;

/// Why a beamformer refused its configuration or its input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeamError {
    /// No beam angle was given.
    NoBeams,
    /// The angle at `index` is not finite or not within `[0, pi]`.
    AngleOutOfRange { index: usize },
    /// The block has `got` microphone channels where the array has
    /// `expected`.
    MicCount { expected: usize, got: usize },
    /// The block has `got` output channels where there are `expected` beams.
    OutputCount { expected: usize, got: usize },
    /// The channels of the block, microphones and outputs, differ in length.
    LengthMismatch,
    /// A microphone sample is NaN or infinite.
    NonFinite,
    /// The direction estimate that drives a selection refused its
    /// configuration.
    Direction(DirectionError),
    /// A selection's `hold` is 0.
    HoldOutOfRange,
    /// A selection's `fade` is 0 or above `hold * hop`.
    FadeOutOfRange { fade: usize },
}

impl fmt::Display for BeamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BeamError::NoBeams => write!(f, "no beam angle was given"),
            BeamError::AngleOutOfRange { index } => write!(
                f,
                "beam angle {index} is not finite and within 0..=pi radians"
            ),
            BeamError::MicCount { expected, got } => write!(
                f,
                "{got} microphone channels where the array has {expected}"
            ),
            BeamError::OutputCount { expected, got } => {
                write!(f, "{got} output channels where there are {expected} beams")
            }
            BeamError::LengthMismatch => write!(f, "the block's channels differ in length"),
            BeamError::NonFinite => write!(f, "a sample is not finite"),
            BeamError::Direction(error) => write!(f, "the direction estimate refused: {error}"),
            BeamError::HoldOutOfRange => write!(f, "the hold is 0"),
            BeamError::FadeOutOfRange { fade } => {
                write!(f, "fade {fade} is not within 1..=hold * hop")
            }
        }
    }
}

impl From<BlockFault> for BeamError {
    fn from(fault: BlockFault) -> BeamError {
        match fault {
            BlockFault::MicCount { expected, got } => BeamError::MicCount { expected, got },
            BlockFault::LengthMismatch => BeamError::LengthMismatch,
            BlockFault::NonFinite => BeamError::NonFinite,
        }
    }
}

impl std::error::Error for BeamError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            BeamError::Direction(error) => Some(error),
            _ => None,
        }
    }
}

/// Delay-and-sum beams at fixed angles, fed in blocks.
///
/// With `M` microphones `spacing` frames apart, the beam steered to angle
/// `phi` delays microphone `m` by
/// `D = lag + (m - (M - 1) / 2) * spacing * cos(phi)` frames and takes the
/// mean of the delayed microphones, so a plane wave from `phi` adds in step
/// and passes at gain 1. Every beam is aligned to the array's centre: a
/// source on a beam's direction comes out as the centre of the array hears
/// it, `lag` frames late.
///
/// **The fractional delay** is a windowed sinc,
/// `h[k] = sinc(k - D) * (0.5 + 0.5 cos(pi (k - D) / 16))` for
/// `|k - D| < 16` and zero elsewhere. With
/// `lag = ceil((M - 1) * spacing / 2) + 16` every filter fits in
/// `2 * lag + 1` taps; for four microphones 1.54 frames apart `lag` is 19.
/// [`FixedBeams::output_lag`] returns it: every beam's output lags the
/// microphones by that many frames, whatever its angle.
///
/// **No state per beam.** A beam is a fixed filter over the microphones'
/// last `2 * lag + 1` frames, kept once for all beams and zero before the
/// start, so moving from one beam's output to another's brings no transient
/// but the move itself.
///
/// **Accuracy and the band.** Up to 0.4 cycles per frame, the response of
/// the beam at `phi` to a plane wave from `theta` at `f` cycles per frame is
/// `|sin(M x / 2) / (M sin(x / 2))|`, `x = 2 pi f spacing (cos theta - cos
/// phi)`, within 0.01 in amplitude. Above 0.45 cycles per frame the
/// fractional delay rolls off and the response falls.
///
/// For a short array that formula says little separates the beams at low
/// frequencies: while `f * M * spacing * |cos theta - cos phi|` is under
/// 0.25, a source is within 1 dB of what it is on its own beam, that is,
/// below `0.25 / (M * spacing * |cos theta - cos phi|)` cycles per frame.
/// For four microphones 1.54 frames apart that is
/// `0.04 / |cos theta - cos phi|`: the broadside beam is near
/// omnidirectional below about 0.04 cycles per frame (650 Hz at 16 kHz),
/// and every beam is near omnidirectional to every source below about 0.02
/// (325 Hz), where a beam on one end of the axis meets a source on the
/// other. A source on the axis is 6 dB down on the broadside beam at 0.10
/// cycles per frame, and its first null there is at 0.16.
///
/// **Cost**: `beams * M * (2 * lag + 1)` multiply-adds per frame.
#[derive(Clone, Debug)]
pub struct FixedBeams {
    mics: usize,
    beams: usize,
    lag: usize,
    /// `2 * lag + 1`.
    taps: usize,
    /// For beam `b` and microphone `m`, the filter divided by `M` and
    /// reversed, at `(b * mics + m) * taps`: entry `j` is
    /// `h[taps - 1 - j] / M`.
    filters: Vec<f32>,
    /// Per microphone, its last `taps - 1` frames, oldest first; during a
    /// block the block follows them.
    lines: Vec<Vec<f32>>,
}

impl FixedBeams {
    /// One beam per angle of `angles`, in that order, every history frame
    /// zero. Refuses an empty `angles` and an angle not finite or outside
    /// `[0, pi]`.
    pub fn new(array: LineArray, angles: &[f32]) -> Result<FixedBeams, BeamError> {
        if angles.is_empty() {
            return Err(BeamError::NoBeams);
        }
        if let Some(index) = angles
            .iter()
            .position(|a| !(0.0..=std::f32::consts::PI).contains(a))
        {
            return Err(BeamError::AngleOutOfRange { index });
        }
        let mics = array.mics();
        let spacing = f64::from(array.spacing());
        let centre = (mics - 1) as f64 / 2.0;
        let lag = (centre * spacing).ceil() as usize + HALF_WIDTH;
        let taps = 2 * lag + 1;
        let mut filters = Vec::with_capacity(angles.len() * mics * taps);
        for &angle in angles {
            let cos = f64::from(angle).cos();
            for m in 0..mics {
                let delay = lag as f64 + (m as f64 - centre) * spacing * cos;
                filters.extend(
                    (0..taps)
                        .rev()
                        .map(|k| (fractional_delay_tap(k as f64 - delay) / mics as f64) as f32),
                );
            }
        }
        Ok(FixedBeams {
            mics,
            beams: angles.len(),
            lag,
            taps,
            filters,
            lines: vec![vec![0.0; taps - 1]; mics],
        })
    }

    /// The count of beams.
    pub fn beams(&self) -> usize {
        self.beams
    }

    /// How many frames every beam's output lags the microphones.
    pub fn output_lag(&self) -> usize {
        self.lag
    }

    /// Writes each beam's output for the block into `out`, one slice per
    /// beam in the order of the angles, continuing from the previous call,
    /// so that any cut of a signal into blocks gives the same output.
    ///
    /// `mics` holds one slice per microphone and `out` one per beam, every
    /// slice of one length. A block with the wrong count of channels,
    /// channels of different lengths or a microphone sample that is not
    /// finite is refused whole, leaving the beams and `out` untouched.
    pub fn process(&mut self, mics: &[&[f32]], out: &mut [&mut [f32]]) -> Result<(), BeamError> {
        let len = check_block(mics, self.mics)?;
        if out.len() != self.beams {
            return Err(BeamError::OutputCount {
                expected: self.beams,
                got: out.len(),
            });
        }
        if out.iter().any(|c| c.len() != len) {
            return Err(BeamError::LengthMismatch);
        }
        self.feed(mics);
        for (b, channel) in out.iter_mut().enumerate() {
            for (t, y) in channel.iter_mut().enumerate() {
                *y = self.sample(b, t);
            }
        }
        self.drop_block(len);
        Ok(())
    }

    /// Appends a checked block to the history lines.
    fn feed(&mut self, mics: &[&[f32]]) {
        for (line, block) in self.lines.iter_mut().zip(mics) {
            line.extend_from_slice(block);
        }
    }

    /// Beam `b`'s output at frame `t` of the block fed last.
    fn sample(&self, b: usize, t: usize) -> f32 {
        let beam = &self.filters[b * self.mics * self.taps..(b + 1) * self.mics * self.taps];
        beam.chunks_exact(self.taps)
            .zip(&self.lines)
            .map(|(filter, line)| {
                filter
                    .iter()
                    .zip(&line[t..t + self.taps])
                    .map(|(h, x)| h * x)
                    .sum::<f32>()
            })
            .sum()
    }

    /// Drops the block of `len` frames fed last from the history lines,
    /// keeping its last `taps - 1` frames.
    fn drop_block(&mut self, len: usize) {
        for line in &mut self.lines {
            line.drain(..len);
        }
    }
}

/// How [`SelectedBeam`] chooses and changes its beam. There is no default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelectionConfig {
    /// The direction estimate that drives the choice. The tested values are
    /// a window of 1024, a hop of 256 and a least peak of 0.2.
    pub direction: DirectionConfig,
    /// How many readings in a row must ask for another beam before the
    /// output moves, at least 1. The tested value is 4.
    pub hold: usize,
    /// The length of the crossfade in frames, at least 1 and at most
    /// `hold * direction.hop`. The tested value is 160.
    pub fade: usize,
}

/// One channel that follows the dominant source from beam to beam, fed in
/// blocks.
///
/// It steers [`FixedBeams`] at the given angles and puts out one of them,
/// chosen by a [`Direction`] estimate over the same microphones. For an array
/// this short the beams differ little where speech has most of its power, so
/// a choice by the beams' output powers would be decided by fractions of a
/// decibel; the phase transform weighs every frequency alike.
///
/// **The rule.** The output starts on the beam nearest `pi / 2`, the lowest
/// index on a tie. A reading with an angle `theta` asks for the beam `k`
/// nearest `theta` (the lowest index on a tie) when `k` is not the current
/// beam `c` and `theta` is nearer to `k`'s angle than to `c`'s by more than a
/// quarter of the angle between the two beams; otherwise it asks for `c`.
/// `hold` readings in a row asking for the same `k` move the output to `k`; a
/// reading asking for `c`, or for a third beam, starts the count again. A
/// reading without an angle changes nothing. The quarter keeps a source
/// midway between two beams from moving the output back and forth.
///
/// **The change.** From the frame of the reading that completes the count,
/// [`crate::doa::Reading::frame`], the output is the old beam's sample times `1 - w`
/// plus the new beam's times `w`, `w` rising linearly from 0 at that frame
/// by `1 / fade` a frame, so the new beam's alone `fade` frames later. Both
/// beams are aligned to the array's centre, so the source is in step in the
/// two. As `fade` is at most `hold * hop`, a fade has always ended before the
/// next can begin.
///
/// **Delays.** The output lags the microphones by
/// [`FixedBeams::output_lag`]. A move of the source is followed within
/// `window + hold * hop` frames, and the fade ends `fade` frames later.
///
/// **Cost per frame**: one beam, two during a fade, at
/// `M * (2 * lag + 1)` multiply-adds each, plus the direction estimate's.
#[derive(Clone, Debug)]
pub struct SelectedBeam {
    beams: FixedBeams,
    angles: Vec<f32>,
    direction: Direction,
    hold: usize,
    fade: usize,
    /// The beam the output is on or moving to.
    selected: usize,
    /// The beam the last change left.
    previous: usize,
    /// The frame from which the last change's fade runs.
    fade_start: u64,
    /// The beam the readings in a row ask for, and how many have.
    asking: Option<(usize, usize)>,
    /// How many frames have been fed.
    fed: u64,
}

impl SelectedBeam {
    /// The selected beam over beams at `angles` for `array`, nothing fed.
    /// Refuses what [`FixedBeams::new`] and [`Direction::new`] refuse, a
    /// `hold` of 0, and a `fade` of 0 or above `hold * hop`.
    pub fn new(
        array: LineArray,
        angles: &[f32],
        config: SelectionConfig,
    ) -> Result<SelectedBeam, BeamError> {
        let beams = FixedBeams::new(array, angles)?;
        let direction = Direction::new(array, config.direction).map_err(BeamError::Direction)?;
        if config.hold == 0 {
            return Err(BeamError::HoldOutOfRange);
        }
        let longest = config.hold.saturating_mul(config.direction.hop);
        if !(1..=longest).contains(&config.fade) {
            return Err(BeamError::FadeOutOfRange { fade: config.fade });
        }
        let half_pi = std::f32::consts::FRAC_PI_2;
        let first = nearest(angles, half_pi);
        Ok(SelectedBeam {
            beams,
            angles: angles.to_vec(),
            direction,
            hold: config.hold,
            fade: config.fade,
            selected: first,
            previous: first,
            fade_start: 0,
            asking: None,
            fed: 0,
        })
    }

    /// How many frames the output lags the microphones.
    pub fn output_lag(&self) -> usize {
        self.beams.output_lag()
    }

    /// The index of the beam the output is on or moving to, in the order of
    /// the angles.
    pub fn selected(&self) -> usize {
        self.selected
    }

    /// Writes the selected beam's output for the block into `out`,
    /// continuing from the previous call, so that any cut of a signal into
    /// blocks gives the same output.
    ///
    /// `mics` holds one slice per microphone, every slice as long as `out`.
    /// A block with the wrong count of channels, channels of different
    /// lengths or a microphone sample that is not finite is refused whole,
    /// leaving the selection and `out` untouched.
    pub fn process(&mut self, mics: &[&[f32]], out: &mut [f32]) -> Result<(), BeamError> {
        let len = check_block(mics, self.beams.mics)?;
        if out.len() != len {
            return Err(BeamError::LengthMismatch);
        }
        let mut readings = Vec::new();
        self.direction.feed(mics, &mut readings);
        self.beams.feed(mics);
        let mut done = 0;
        for reading in &readings {
            let until = (reading.frame - self.fed) as usize;
            self.fill(&mut out[done..until], done);
            done = until;
            if let Some(angle) = reading.angle {
                self.vote(angle, reading.frame);
            }
        }
        self.fill(&mut out[done..], done);
        self.beams.drop_block(len);
        self.fed += len as u64;
        Ok(())
    }

    /// Writes the output for the frames of the block fed last from `offset`
    /// on into `out`.
    fn fill(&self, out: &mut [f32], offset: usize) {
        for (i, y) in out.iter_mut().enumerate() {
            let t = offset + i;
            let frame = self.fed + t as u64;
            let into_fade = frame - self.fade_start;
            *y = if self.previous == self.selected || into_fade >= self.fade as u64 {
                self.beams.sample(self.selected, t)
            } else {
                let w = into_fade as f32 / self.fade as f32;
                self.beams.sample(self.previous, t) * (1.0 - w)
                    + self.beams.sample(self.selected, t) * w
            };
        }
    }

    /// Counts the vote of a reading of `angle` made at `frame`, moving the
    /// output when it completes `hold` in a row.
    fn vote(&mut self, angle: f32, frame: u64) {
        let current = self.selected;
        let k = nearest(&self.angles, angle);
        let (to_k, to_current) = (
            (angle - self.angles[k]).abs(),
            (angle - self.angles[current]).abs(),
        );
        let between = (self.angles[k] - self.angles[current]).abs();
        if k == current || to_current - to_k <= between / 4.0 {
            self.asking = None;
            return;
        }
        let count = match self.asking {
            Some((beam, count)) if beam == k => count + 1,
            _ => 1,
        };
        if count < self.hold {
            self.asking = Some((k, count));
            return;
        }
        self.asking = None;
        self.previous = current;
        self.selected = k;
        self.fade_start = frame;
    }
}

/// The index of the angle of `angles` nearest `angle`, the lowest on a tie.
fn nearest(angles: &[f32], angle: f32) -> usize {
    (1..angles.len()).fold(0, |best, i| {
        if (angles[i] - angle).abs() < (angles[best] - angle).abs() {
            i
        } else {
            best
        }
    })
}

/// The windowed sinc at `u` frames from its centre:
/// `sinc(u) * (0.5 + 0.5 cos(pi u / 16))` for `|u| < 16`, zero elsewhere.
fn fractional_delay_tap(u: f64) -> f64 {
    let half = HALF_WIDTH as f64;
    if u.abs() >= half {
        return 0.0;
    }
    let sinc = if u == 0.0 {
        1.0
    } else {
        (PI * u).sin() / (PI * u)
    };
    sinc * (0.5 + 0.5 * (PI * u / half).cos())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_signals::{
        DEGREE, amplitude, bits, low_passed, mean_square, plane_tone, plane_wave, power_db, sum,
        test_array, white, with_noise,
    };

    fn degrees(angles: &[f32]) -> Vec<f32> {
        angles.iter().map(|a| a * DEGREE).collect()
    }

    /// Every beam's output for `mics` fed as one block to fresh beams.
    fn beam_outputs(angles: &[f32], mics: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut beams = FixedBeams::new(test_array(), angles).unwrap();
        let inputs: Vec<&[f32]> = mics.iter().map(Vec::as_slice).collect();
        let mut out = vec![vec![0.0; mics[0].len()]; angles.len()];
        let mut slices: Vec<&mut [f32]> = out.iter_mut().map(Vec::as_mut_slice).collect();
        beams.process(&inputs, &mut slices).unwrap();
        out
    }

    /// The formula's amplitude response of the beam at `phi` to a plane
    /// wave from `theta` at `f` cycles per frame.
    fn formula(f: f64, theta: f64, phi: f64) -> f64 {
        let (mics, spacing) = (4.0, 1.54f32);
        let x = 2.0 * PI * f * f64::from(spacing) * (theta.cos() - phi.cos());
        let below = mics * (x / 2.0).sin();
        if below.abs() < 1e-9 {
            1.0
        } else {
            ((mics * x / 2.0).sin() / below).abs()
        }
    }

    #[test]
    fn the_lag_is_the_half_line_rounded_up_plus_the_half_filter() {
        let beams = FixedBeams::new(test_array(), &degrees(&[30.0, 90.0])).unwrap();
        assert_eq!(beams.output_lag(), 19);
        assert_eq!(beams.beams(), 2);
        let long = LineArray::new(2, 1024.0).unwrap();
        assert_eq!(FixedBeams::new(long, &[0.0]).unwrap().output_lag(), 528);
    }

    #[test]
    fn a_beam_follows_its_formula() {
        let angles = degrees(&[0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]);
        for f in [0.02, 0.05, 0.1, 0.2, 0.3, 0.4] {
            for &theta in &angles {
                let mics = plane_tone(&test_array(), theta, f, 1500);
                let out = beam_outputs(&angles, &mics);
                for (&phi, channel) in angles.iter().zip(&out) {
                    let got = amplitude(channel, f, 39);
                    let want = formula(f, f64::from(theta), f64::from(phi));
                    assert!(
                        (got - want).abs() <= 0.01,
                        "f {f}, source {theta}, beam {phi}: got {got:.4}, want {want:.4}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_beam_favours_its_direction_over_independent_noise() {
        let angles = degrees(&[30.0, 60.0, 90.0, 120.0, 150.0]);
        let len = 16384;
        let noise: Vec<Vec<f32>> = (0..4).map(|m| white(len, 0.1, 401 + m)).collect();
        let noise_out = beam_outputs(&angles, &noise);
        for (b, &angle) in angles.iter().enumerate() {
            let wave = plane_wave(&test_array(), angle, (0.0, 0.5), len, 0.1, 411);
            let out = beam_outputs(&angles, &wave.mics);
            let before = power_db(&wave.mics[0]) - power_db(&noise[0]);
            let after = power_db(&out[b][39..]) - power_db(&noise_out[b][39..]);
            let gain = after - before;
            assert!(gain >= 5.5, "beam at {angle}: {gain:.2} dB");
        }
    }

    #[test]
    fn the_broadside_beam_rejects_a_source_on_the_axis_above_the_low_band() {
        for axis in [0.0, std::f32::consts::PI] {
            let angles = [axis, std::f32::consts::FRAC_PI_2];
            let wave = plane_wave(&test_array(), axis, (0.1, 0.4), 16384, 0.1, 421);
            let out = beam_outputs(&angles, &wave.mics);
            let rejection = power_db(&out[0][39..]) - power_db(&out[1][39..]);
            assert!(rejection >= 8.0, "axis {axis}: {rejection:.2} dB");
        }
    }

    #[test]
    fn every_beam_is_near_omnidirectional_low_down() {
        for axis in [0.0, std::f32::consts::PI] {
            let angles = [axis, std::f32::consts::FRAC_PI_2];
            let mics = plane_tone(&test_array(), axis, 0.02, 1500);
            let out = beam_outputs(&angles, &mics);
            let own = amplitude(&out[0], 0.02, 39);
            let broadside = amplitude(&out[1], 0.02, 39);
            let difference = 20.0 * (own / broadside).log10();
            assert!(difference.abs() <= 1.0, "axis {axis}: {difference:.2} dB");
        }
    }

    #[test]
    fn a_source_on_a_beam_comes_out_late_by_the_output_lag() {
        let angles = degrees(&[0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]);
        let lag = FixedBeams::new(test_array(), &angles).unwrap().output_lag();
        for (b, &angle) in angles.iter().enumerate() {
            let wave = plane_wave(&test_array(), angle, (0.0, 0.4), 8192, 0.1, 431);
            let out = beam_outputs(&angles, &wave.mics);
            let residual: Vec<f32> = out[b][2 * lag..]
                .iter()
                .zip(&wave.centre[lag..])
                .map(|(y, s)| y - s)
                .collect();
            let below = 10.0 * (mean_square(&wave.centre) / mean_square(&residual)).log10();
            assert!(below >= 60.0, "beam at {angle}: {below:.1} dB");
        }
    }

    #[test]
    fn any_cut_into_blocks_gives_the_same_output() {
        let angles = degrees(&[30.0, 60.0, 90.0, 120.0, 150.0]);
        let len = 3000;
        let wave = plane_wave(&test_array(), 50.0 * DEGREE, (0.0, 0.5), 4096, 0.1, 441);
        let mics: Vec<Vec<f32>> = wave
            .mics
            .iter()
            .enumerate()
            .map(|(m, c)| {
                let noise = white(len, 0.01, 451 + m as u64);
                c[..len].iter().zip(&noise).map(|(x, n)| x + n).collect()
            })
            .collect();
        let whole = beam_outputs(&angles, &mics);
        for block in [1, 7, 160] {
            let mut beams = FixedBeams::new(test_array(), &angles).unwrap();
            let mut out = vec![vec![0.0; len]; angles.len()];
            for start in (0..len).step_by(block) {
                let end = (start + block).min(len);
                let inputs: Vec<&[f32]> = mics.iter().map(|c| &c[start..end]).collect();
                let mut slices: Vec<&mut [f32]> =
                    out.iter_mut().map(|c| &mut c[start..end]).collect();
                beams.process(&inputs, &mut slices).unwrap();
            }
            for (b, (got, want)) in out.iter().zip(&whole).enumerate() {
                assert_eq!(bits(got), bits(want), "blocks of {block}, beam {b}");
            }
        }
    }

    #[test]
    fn a_bad_configuration_is_refused() {
        assert_eq!(
            FixedBeams::new(test_array(), &[]).err(),
            Some(BeamError::NoBeams)
        );
        for bad in [f32::NAN, f32::INFINITY, -0.01, 3.15] {
            assert_eq!(
                FixedBeams::new(test_array(), &[0.5, 1.0, bad]).err(),
                Some(BeamError::AngleOutOfRange { index: 2 }),
                "angle {bad}"
            );
        }
        assert!(FixedBeams::new(test_array(), &[0.0, std::f32::consts::PI]).is_ok());
    }

    #[test]
    fn a_bad_block_is_refused_and_changes_nothing() {
        let angles = degrees(&[60.0, 120.0]);
        let first: Vec<Vec<f32>> = (0..4).map(|m| white(100, 0.1, 461 + m)).collect();
        let second: Vec<Vec<f32>> = (0..4).map(|m| white(100, 0.1, 471 + m)).collect();
        let feed = |beams: &mut FixedBeams, mics: &[Vec<f32>]| -> Vec<Vec<f32>> {
            let inputs: Vec<&[f32]> = mics.iter().map(Vec::as_slice).collect();
            let mut out = vec![vec![0.0; 100]; 2];
            let mut slices: Vec<&mut [f32]> = out.iter_mut().map(Vec::as_mut_slice).collect();
            beams.process(&inputs, &mut slices).unwrap();
            out
        };
        let mut clean = FixedBeams::new(test_array(), &angles).unwrap();
        let mut refused = FixedBeams::new(test_array(), &angles).unwrap();
        feed(&mut clean, &first);
        feed(&mut refused, &first);

        let good: Vec<&[f32]> = second.iter().map(Vec::as_slice).collect();
        let short = &second[3][..99];
        let mut nan = second[2].clone();
        nan[50] = f32::NAN;
        let mut infinite = second[1].clone();
        infinite[0] = f32::NEG_INFINITY;
        let cases: Vec<(Vec<&[f32]>, usize, usize, BeamError)> = vec![
            (
                good[..3].to_vec(),
                2,
                100,
                BeamError::MicCount {
                    expected: 4,
                    got: 3,
                },
            ),
            (
                good.clone(),
                3,
                100,
                BeamError::OutputCount {
                    expected: 2,
                    got: 3,
                },
            ),
            (
                vec![good[0], good[1], good[2], short],
                2,
                100,
                BeamError::LengthMismatch,
            ),
            (good.clone(), 2, 99, BeamError::LengthMismatch),
            (
                vec![good[0], good[1], &nan, good[3]],
                2,
                100,
                BeamError::NonFinite,
            ),
            (
                vec![good[0], &infinite, good[2], good[3]],
                2,
                100,
                BeamError::NonFinite,
            ),
        ];
        for (inputs, outputs, out_len, error) in cases {
            let mut out = vec![vec![7.0f32; out_len]; outputs];
            let mut slices: Vec<&mut [f32]> = out.iter_mut().map(Vec::as_mut_slice).collect();
            assert_eq!(refused.process(&inputs, &mut slices), Err(error));
            assert!(
                out.iter().flatten().all(|&y| y == 7.0),
                "{error:?}: the output was touched"
            );
        }
        let want = feed(&mut clean, &second);
        let got = feed(&mut refused, &second);
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(bits(g), bits(w));
        }
    }

    const SELECTION: SelectionConfig = SelectionConfig {
        direction: DirectionConfig {
            window: 1024,
            hop: 256,
            min_peak: 0.2,
        },
        hold: 4,
        fade: 160,
    };

    /// `window + hold * hop`: by this frame a steady source has been
    /// followed.
    const SETTLED: usize = 1024 + 4 * 256;

    /// The index of the broadside beam among [`selection_angles`].
    const BROADSIDE: usize = 2;

    fn selection_angles() -> Vec<f32> {
        degrees(&[30.0, 60.0, 90.0, 120.0, 150.0])
    }

    /// The selected beam's output for `mics` fed in blocks of `block`, and
    /// after each block the count of frames fed and the beam selected.
    fn select(mics: &[Vec<f32>], block: usize) -> (Vec<f32>, Vec<(usize, usize)>) {
        let mut selected = SelectedBeam::new(test_array(), &selection_angles(), SELECTION).unwrap();
        let len = mics[0].len();
        let mut out = vec![0.0; len];
        let mut trace = Vec::new();
        for start in (0..len).step_by(block) {
            let end = (start + block).min(len);
            let inputs: Vec<&[f32]> = mics.iter().map(|c| &c[start..end]).collect();
            selected.process(&inputs, &mut out[start..end]).unwrap();
            trace.push((end, selected.selected()));
        }
        (out, trace)
    }

    /// A white source at 10 dB over independent noise that jumps from 60
    /// degrees to 120 at frame `JUMP`.
    fn jumping_source() -> Vec<Vec<f32>> {
        let len = 8192;
        let before = plane_wave(&test_array(), 60.0 * DEGREE, (0.0, 0.5), len, 0.1, 701);
        let after = plane_wave(&test_array(), 120.0 * DEGREE, (0.0, 0.5), len, 0.1, 711);
        let mics: Vec<Vec<f32>> = before
            .mics
            .iter()
            .zip(&after.mics)
            .map(|(a, b)| a[..JUMP].iter().chain(&b[JUMP..]).copied().collect())
            .collect();
        with_noise(&mics, 0.01, 721)
    }

    const JUMP: usize = 4096;

    #[test]
    fn the_selection_starts_on_the_beam_nearest_broadside() {
        let config = SELECTION;
        let pick = |angles: &[f32]| {
            SelectedBeam::new(test_array(), angles, config)
                .unwrap()
                .selected()
        };
        assert_eq!(pick(&selection_angles()), 2);
        assert_eq!(pick(&degrees(&[150.0, 100.0, 10.0])), 1);
        // Both a quarter turn from broadside: the tie goes to the lower index.
        assert_eq!(pick(&[std::f32::consts::PI, 0.0]), 0);
        assert_eq!(pick(&[0.0]), 0);
    }

    #[test]
    fn a_steady_source_is_followed_and_held() {
        let angles = selection_angles();
        for (b, &beam) in angles.iter().enumerate() {
            for off in [0.0, 5.0, -5.0] {
                let source = beam + off * DEGREE;
                let wave = plane_wave(&test_array(), source, (0.0, 0.5), 8192, 0.1, 731 + b as u64);
                let mics = with_noise(&wave.mics, 0.01, 741);
                let (out, trace) = select(&mics, 128);
                for &(fed, selected) in &trace {
                    let allowed = fed >= SETTLED && selected == b
                        || fed < SETTLED && (selected == b || selected == BROADSIDE);
                    assert!(allowed, "beam {b}, off {off}: beam {selected} at {fed}");
                }
                let fixed = beam_outputs(&angles, &mics);
                let from = if b == BROADSIDE { 0 } else { SETTLED + 160 };
                assert_eq!(
                    bits(&out[from..]),
                    bits(&fixed[b][from..]),
                    "beam {b}, off {off}"
                );
            }
        }
    }

    #[test]
    fn a_source_between_two_beams_leaves_the_output_where_it_is() {
        let wave = plane_wave(&test_array(), 75.0 * DEGREE, (0.0, 0.5), 8192, 0.1, 751);
        let (_, trace) = select(&with_noise(&wave.mics, 0.01, 761), 128);
        assert!(trace.iter().all(|&(_, s)| s == BROADSIDE), "{trace:?}");
    }

    #[test]
    fn a_jump_of_the_source_is_followed_with_a_smooth_fade() {
        let angles = selection_angles();
        let mics = jumping_source();
        let (out, trace) = select(&mics, 1);
        let fixed = beam_outputs(&angles, &mics);
        for &(fed, selected) in &trace {
            if (SETTLED..=JUMP).contains(&fed) {
                assert_eq!(selected, 1, "before the jump, at {fed}");
            }
        }
        let start = trace
            .iter()
            .find(|&&(fed, s)| fed > JUMP && s == 3)
            .map(|&(fed, _)| fed)
            .expect("the jump was never followed");
        assert!(start <= JUMP + SETTLED, "followed at {start}");
        for &(fed, selected) in &trace {
            if fed > JUMP {
                let want = if fed < start { 1 } else { 3 };
                assert_eq!(selected, want, "after the jump, at {fed}");
            }
        }
        let largest_step = |signal: &[f32]| {
            (start..=start + 160)
                .map(|t| (signal[t] - signal[t - 1]).abs())
                .fold(0.0, f32::max)
        };
        let fade = largest_step(&out);
        let beams = largest_step(&fixed[1]).max(largest_step(&fixed[3]));
        assert!(fade <= 1.1 * beams, "fade {fade}, beams {beams}");
        assert_eq!(
            bits(&out[SETTLED + 160..start]),
            bits(&fixed[1][SETTLED + 160..start])
        );
        assert_eq!(bits(&out[start + 160..]), bits(&fixed[3][start + 160..]));
    }

    #[test]
    fn silence_and_independent_noise_leave_the_first_beam() {
        let len = 8192;
        let zeros = vec![vec![0.0; len]; 4];
        let whites: Vec<Vec<f32>> = (0..4).map(|m| white(len, 0.1, 771 + m)).collect();
        let lows: Vec<Vec<f32>> = (0..4).map(|m| low_passed(len, 0.1, 781 + m)).collect();
        for (name, mics) in [("silence", zeros), ("white", whites), ("low-passed", lows)] {
            let (_, trace) = select(&mics, 256);
            assert!(trace.iter().all(|&(_, s)| s == BROADSIDE), "{name}");
        }
    }

    #[test]
    fn a_weaker_second_source_does_not_move_the_selection() {
        for (strong, weak, beam) in [(60.0, 120.0, 1), (150.0, 30.0, 4), (90.0, 30.0, 2)] {
            let len = 8192;
            let a = plane_wave(&test_array(), strong * DEGREE, (0.0, 0.5), len, 0.1, 791);
            let b = plane_wave(&test_array(), weak * DEGREE, (0.0, 0.5), len, 0.025, 801);
            let both: Vec<Vec<f32>> = a.mics.iter().zip(&b.mics).map(|(x, y)| sum(x, y)).collect();
            let (_, trace) = select(&with_noise(&both, 0.01, 811), 128);
            for &(fed, selected) in &trace {
                let allowed = selected == beam || fed < SETTLED && selected == BROADSIDE;
                assert!(allowed, "{strong} over {weak}: beam {selected} at {fed}");
            }
        }
    }

    #[test]
    fn hold_readings_in_a_row_move_the_output_and_others_restart_the_count() {
        let angles = selection_angles();
        let mut beam = SelectedBeam::new(test_array(), &angles, SELECTION).unwrap();
        let mut vote = |degrees: f32| {
            beam.vote(degrees * DEGREE, 0);
            beam.selected()
        };
        // Three for 30, one for 90 (the current beam), three for 30.
        for angle in [30.0, 30.0, 30.0, 90.0, 30.0, 30.0, 30.0] {
            assert_eq!(vote(angle), BROADSIDE);
        }
        // One for 150, a third beam, then three for 30.
        for angle in [150.0, 30.0, 30.0, 30.0] {
            assert_eq!(vote(angle), BROADSIDE);
        }
        // 80 is nearest 90; 72 is nearest 60 but within the quarter; then
        // four for 30.
        for angle in [80.0, 72.0, 72.0, 72.0, 72.0, 30.0, 30.0, 30.0] {
            assert_eq!(vote(angle), BROADSIDE);
        }
        assert_eq!(vote(30.0), 0);
        // After the move the count starts afresh.
        for angle in [150.0, 150.0, 150.0] {
            assert_eq!(vote(angle), 0);
        }
        assert_eq!(vote(150.0), 4);
    }

    #[test]
    fn any_cut_into_blocks_gives_the_same_selected_output() {
        let mics = jumping_source();
        let (whole, whole_trace) = select(&mics, mics[0].len());
        for block in [1, 7, 160] {
            let (out, trace) = select(&mics, block);
            assert_eq!(bits(&out), bits(&whole), "blocks of {block}");
            assert_eq!(trace.last(), whole_trace.last(), "blocks of {block}");
        }
    }

    #[test]
    fn a_bad_selection_is_refused() {
        let new = |angles: &[f32], config| SelectedBeam::new(test_array(), angles, config).err();
        let angles = selection_angles();
        assert_eq!(new(&[], SELECTION), Some(BeamError::NoBeams));
        assert_eq!(
            new(&[0.5, f32::NAN], SELECTION),
            Some(BeamError::AngleOutOfRange { index: 1 })
        );
        let mut config = SELECTION;
        config.direction.window = 1000;
        assert_eq!(
            new(&angles, config),
            Some(BeamError::Direction(DirectionError::WindowOutOfRange {
                window: 1000
            }))
        );
        config = SELECTION;
        config.hold = 0;
        assert_eq!(new(&angles, config), Some(BeamError::HoldOutOfRange));
        for fade in [0, 1025] {
            config = SELECTION;
            config.fade = fade;
            assert_eq!(
                new(&angles, config),
                Some(BeamError::FadeOutOfRange { fade })
            );
        }
        config = SELECTION;
        config.fade = 1024;
        assert_eq!(new(&angles, config), None);
    }

    #[test]
    fn a_bad_block_is_refused_and_changes_no_selection() {
        let wave = plane_wave(&test_array(), 30.0 * DEGREE, (0.0, 0.5), 4096, 0.1, 821);
        let mics = with_noise(&wave.mics, 0.01, 831);
        let (want, want_trace) = select(&mics, 1000);
        let mut beam = SelectedBeam::new(test_array(), &selection_angles(), SELECTION).unwrap();
        let mut got = vec![0.0; 4096];
        let block = |start: usize, end: usize| -> Vec<&[f32]> {
            mics.iter().map(|c| &c[start..end]).collect()
        };
        beam.process(&block(0, 1000), &mut got[..1000]).unwrap();

        let good = block(1000, 2000);
        let short = &mics[3][1000..1999];
        let mut nan = mics[2][1000..2000].to_vec();
        nan[50] = f32::NAN;
        let cases: Vec<(Vec<&[f32]>, usize, BeamError)> = vec![
            (
                good[..3].to_vec(),
                1000,
                BeamError::MicCount {
                    expected: 4,
                    got: 3,
                },
            ),
            (
                vec![good[0], good[1], good[2], short],
                1000,
                BeamError::LengthMismatch,
            ),
            (good.clone(), 999, BeamError::LengthMismatch),
            (
                vec![good[0], good[1], &nan, good[3]],
                1000,
                BeamError::NonFinite,
            ),
        ];
        for (inputs, out_len, error) in cases {
            let mut out = vec![7.0f32; out_len];
            assert_eq!(beam.process(&inputs, &mut out), Err(error));
            assert!(out.iter().all(|&y| y == 7.0), "{error:?}: output touched");
        }
        for start in (1000..4096).step_by(1000) {
            let end = (start + 1000).min(4096);
            beam.process(&block(start, end), &mut got[start..end])
                .unwrap();
        }
        assert_eq!(bits(&got), bits(&want));
        assert_eq!(Some(beam.selected()), want_trace.last().map(|&(_, s)| s));
        assert_eq!(beam.selected(), 0);
    }
}
