//! A streaming high-pass filter.
//!
//! [`HighPass`] is a fourth-order Butterworth high-pass, the front end an
//! echo canceller wants on real microphones: the same filter on the
//! reference and on each microphone leaves the echo path as it was and keeps
//! low-frequency power that carries no echo out of the canceller.

use std::f64::consts::PI;
use std::fmt;

/// The lowest and highest cutoffs accepted, in cycles per frame: the range
/// over which the filter's `f32` response was checked against its formula.
const MIN_CUTOFF: f32 = 0.001;
const MAX_CUTOFF: f32 = 0.4;

/// Why a [`HighPass`] refused its input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterError {
    /// The cutoff is not within 0.001..=0.4 cycles per frame.
    CutoffOutOfRange,
    /// A sample is NaN or infinite.
    NonFinite,
}

impl fmt::Display for FilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FilterError::CutoffOutOfRange => write!(
                f,
                "the cutoff is not within {MIN_CUTOFF}..={MAX_CUTOFF} cycles per frame"
            ),
            FilterError::NonFinite => write!(f, "a sample is not finite"),
        }
    }
}

impl std::error::Error for FilterError {}

/// One second-order section in transposed direct form II. The numerator is
/// `gain (1 - 2 z^-1 + z^-2)`.
#[derive(Clone, Debug)]
struct Section {
    gain: f32,
    a1: f32,
    a2: f32,
    s1: f32,
    s2: f32,
}

impl Section {
    /// The high-pass section of quality `quality` at the pre-warped cutoff
    /// `warped = tan(pi cutoff)`, by the bilinear transform.
    fn new(warped: f64, quality: f64) -> Section {
        let square = warped * warped;
        let norm = 1.0 / (1.0 + warped / quality + square);
        Section {
            gain: norm as f32,
            a1: (2.0 * (square - 1.0) * norm) as f32,
            a2: ((1.0 - warped / quality + square) * norm) as f32,
            s1: 0.0,
            s2: 0.0,
        }
    }

    fn step(&mut self, x: f32) -> f32 {
        let y = self.gain * x + self.s1;
        self.s1 = -2.0 * self.gain * x - self.a1 * y + self.s2;
        self.s2 = self.gain * x - self.a2 * y;
        y
    }
}

/// A fourth-order Butterworth high-pass, fed in blocks.
///
/// Two second-order sections designed by the bilinear transform with the
/// cutoff pre-warped. The power response is
/// `1 / (1 + (tan(pi cutoff) / tan(pi f))^8)` at `f` cycles per frame:
/// -3.01 dB at the cutoff, -24.1 dB an octave below. The filter works in the
/// caller's units and is meant for samples of full scale about 1; a sample
/// near `f32::MAX` can overflow inside it, and the output is then not
/// finite.
#[derive(Clone, Debug)]
pub struct HighPass {
    sections: [Section; 2],
}

impl HighPass {
    /// A filter at `cutoff` cycles per frame, within `0.001..=0.4`: 200 Hz at
    /// 16 kHz is `0.0125`. Every state variable starts at zero.
    pub fn new(cutoff: f32) -> Result<HighPass, FilterError> {
        if !(MIN_CUTOFF..=MAX_CUTOFF).contains(&cutoff) {
            return Err(FilterError::CutoffOutOfRange);
        }
        let warped = (PI * f64::from(cutoff)).tan();
        Ok(HighPass {
            sections: [
                Section::new(warped, 1.0 / (2.0 * (PI / 8.0).cos())),
                Section::new(warped, 1.0 / (2.0 * (3.0 * PI / 8.0).cos())),
            ],
        })
    }

    /// Filters `samples` in place, continuing from the previous call, so that
    /// any cut of a signal into blocks gives the same output. A block holding
    /// a sample that is not finite is refused whole, leaving the filter and
    /// the samples untouched.
    pub fn process(&mut self, samples: &mut [f32]) -> Result<(), FilterError> {
        if !samples.iter().all(|x| x.is_finite()) {
            return Err(FilterError::NonFinite);
        }
        for x in samples.iter_mut() {
            let [first, second] = &mut self.sections;
            *x = second.step(first.step(*x));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_signals::{bits, white};

    /// The formula's gain in dB at `frequency` for a filter at `cutoff`.
    fn formula_db(cutoff: f64, frequency: f64) -> f64 {
        let ratio = (PI * cutoff).tan() / (PI * frequency).tan();
        -10.0 * (1.0 + ratio.powi(8)).log10()
    }

    /// The amplitude of the sinusoid at `frequency` in `signal`, fitted by
    /// least squares on a sine and a cosine starting at frame `start`.
    fn amplitude(signal: &[f32], frequency: f64, start: usize) -> f64 {
        let (mut ss, mut sc, mut cc, mut ys, mut yc) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (t, &y) in signal.iter().enumerate().skip(start) {
            let (s, c) = (2.0 * PI * frequency * t as f64).sin_cos();
            let y = f64::from(y);
            ss += s * s;
            sc += s * c;
            cc += c * c;
            ys += y * s;
            yc += y * c;
        }
        let det = ss * cc - sc * sc;
        let a = (ys * cc - yc * sc) / det;
        let b = (yc * ss - ys * sc) / det;
        a.hypot(b)
    }

    #[test]
    fn the_response_follows_the_formula() {
        for cutoff in [0.001f32, 0.0125, 0.1, 0.4] {
            let c = f64::from(cutoff);
            // Long enough for the slowest transient to fall far below the
            // quietest output, then four periods of the lowest frequency.
            let settle = (40.0 / c) as usize;
            let len = settle + (16.0 / c) as usize;
            for factor in [0.25, 0.5, 1.0, 2.0, 4.0] {
                let frequency = c * factor;
                if frequency >= 0.5 {
                    continue;
                }
                let mut signal: Vec<f32> = (0..len)
                    .map(|t| (2.0 * PI * frequency * t as f64).sin() as f32)
                    .collect();
                let mut filter = HighPass::new(cutoff).unwrap();
                filter.process(&mut signal).unwrap();
                let got = 20.0 * amplitude(&signal, frequency, settle).log10();
                let want = formula_db(c, frequency);
                assert!(
                    (got - want).abs() <= 0.1,
                    "cutoff {cutoff}, {factor} times: got {got:.3} dB, want {want:.3} dB"
                );
            }
        }
    }

    #[test]
    fn a_constant_is_removed() {
        let mut signal = vec![1.0f32; 4000];
        HighPass::new(0.0125).unwrap().process(&mut signal).unwrap();
        for (t, y) in signal.iter().enumerate().skip(2000) {
            assert!(y.abs() < 1e-6, "frame {t}: {y}");
        }
    }

    #[test]
    fn any_cut_into_blocks_gives_the_same_output() {
        let input = white(5000, 0.1, 301);
        let mut whole = input.clone();
        HighPass::new(0.0125).unwrap().process(&mut whole).unwrap();
        for block in [1, 7, 160] {
            let mut cut = input.clone();
            let mut filter = HighPass::new(0.0125).unwrap();
            for chunk in cut.chunks_mut(block) {
                filter.process(chunk).unwrap();
            }
            assert_eq!(bits(&cut), bits(&whole), "blocks of {block}");
        }
    }

    #[test]
    fn a_cutoff_out_of_range_is_refused() {
        for cutoff in [0.0, 0.0009, 0.41, 0.5, -0.0125, f32::NAN, f32::INFINITY] {
            assert_eq!(
                HighPass::new(cutoff).err(),
                Some(FilterError::CutoffOutOfRange),
                "cutoff {cutoff}"
            );
        }
    }

    #[test]
    fn a_block_with_a_non_finite_sample_is_refused_and_changes_nothing() {
        let first = white(300, 0.1, 311);
        let second = white(300, 0.1, 312);
        let mut clean = HighPass::new(0.0125).unwrap();
        let mut refused = HighPass::new(0.0125).unwrap();
        clean.process(&mut first.clone()).unwrap();
        refused.process(&mut first.clone()).unwrap();
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut block = second.clone();
            block[150] = bad;
            let before = block.clone();
            assert_eq!(refused.process(&mut block), Err(FilterError::NonFinite));
            assert_eq!(bits(&block), bits(&before), "{bad}: the block was touched");
        }
        let mut expected = second.clone();
        clean.process(&mut expected).unwrap();
        let mut got = second;
        refused.process(&mut got).unwrap();
        assert_eq!(bits(&got), bits(&expected));
    }
}
