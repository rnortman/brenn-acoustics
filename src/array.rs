//! A line of microphones, and the conventions every multi-microphone
//! algorithm in the crate shares.
//!
//! **Channels.** Several microphones are passed as `&[&[f32]]`: one slice per
//! microphone, in the order the microphones lie along the line, all of one
//! length.
//!
//! **The angle.** A direction is the angle from the array's axis, in
//! radians, within `[0, pi]`. Angle zero is the direction from the first
//! microphone towards the last: a source at angle 0 lies on the axis beyond
//! the last microphone and its sound reaches the last microphone first.
//! `pi / 2` is broadside.
//!
//! **What a line cannot tell.** A line gives the angle from its axis and
//! nothing more: every direction on the cone at that angle, both sides of
//! the line included, is the same to it.
//!
//! **Far field.** A source is taken to be far enough that its wavefront is
//! plane: microphone `m` hears it `m * spacing * cos(angle)` frames before
//! microphone 0.

use std::fmt;

/// The fewest and most microphones a [`LineArray`] accepts.
const MIN_MICS: usize = 2;
const MAX_MICS: usize = 64;
/// The longest line accepted, first microphone to last, in frames.
const MAX_LENGTH: f32 = 1024.0;

/// Why a [`LineArray`] refused its geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrayError {
    /// The count of microphones is not within 2..=64.
    MicsOutOfRange { mics: usize },
    /// The spacing is not finite and above zero.
    SpacingOutOfRange,
    /// The line is longer than 1024 frames from its first microphone to its
    /// last.
    TooLong,
}

impl fmt::Display for ArrayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArrayError::MicsOutOfRange { mics } => write!(
                f,
                "{mics} microphones is not within {MIN_MICS}..={MAX_MICS}"
            ),
            ArrayError::SpacingOutOfRange => {
                write!(f, "the spacing is not finite and above zero")
            }
            ArrayError::TooLong => write!(
                f,
                "the line is longer than {MAX_LENGTH} frames from end to end"
            ),
        }
    }
}

impl std::error::Error for ArrayError {}

/// Microphones evenly spaced on a line.
///
/// `spacing` is the time sound takes from one microphone to the next along
/// the line, in frames: 33 mm at 16 kHz is 1.54. The caps on the count and on
/// the line's length bound every allocation the crate makes from an array.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineArray {
    mics: usize,
    spacing: f32,
}

impl LineArray {
    /// `mics` microphones `spacing` frames apart. Refuses fewer than 2 or
    /// more than 64 microphones, a spacing that is not finite and above
    /// zero, and a line longer than 1024 frames end to end.
    pub fn new(mics: usize, spacing: f32) -> Result<LineArray, ArrayError> {
        if !(MIN_MICS..=MAX_MICS).contains(&mics) {
            return Err(ArrayError::MicsOutOfRange { mics });
        }
        if !(spacing.is_finite() && spacing > 0.0) {
            return Err(ArrayError::SpacingOutOfRange);
        }
        if f64::from(spacing) * (mics - 1) as f64 > f64::from(MAX_LENGTH) {
            return Err(ArrayError::TooLong);
        }
        Ok(LineArray { mics, spacing })
    }

    /// The count of microphones.
    pub fn mics(&self) -> usize {
        self.mics
    }

    /// The time from one microphone to the next, in frames.
    pub fn spacing(&self) -> f32 {
        self.spacing
    }
}

/// How a block of microphone channels fails the channel convention. Each
/// module's error enum takes it into its own variants of the same names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlockFault {
    MicCount { expected: usize, got: usize },
    LengthMismatch,
    NonFinite,
}

/// The length of `mics`, a block for `expected` microphones. Refuses the
/// wrong count of channels, channels of different lengths and a sample that
/// is not finite, in that order.
pub(crate) fn check_block(mics: &[&[f32]], expected: usize) -> Result<usize, BlockFault> {
    if mics.len() != expected {
        return Err(BlockFault::MicCount {
            expected,
            got: mics.len(),
        });
    }
    let len = mics[0].len();
    if mics.iter().any(|c| c.len() != len) {
        return Err(BlockFault::LengthMismatch);
    }
    if !mics.iter().all(|c| c.iter().all(|x| x.is_finite())) {
        return Err(BlockFault::NonFinite);
    }
    Ok(len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_array_keeps_its_geometry() {
        let array = LineArray::new(4, 1.54).unwrap();
        assert_eq!(array.mics(), 4);
        assert_eq!(array.spacing(), 1.54);
        assert!(LineArray::new(2, 1024.0).is_ok());
        assert!(LineArray::new(64, 16.0).is_ok());
    }

    #[test]
    fn a_count_out_of_range_is_refused() {
        for mics in [0, 1, 65, usize::MAX] {
            assert_eq!(
                LineArray::new(mics, 1.0).err(),
                Some(ArrayError::MicsOutOfRange { mics }),
                "{mics} microphones"
            );
        }
    }

    #[test]
    fn a_spacing_out_of_range_is_refused() {
        for spacing in [0.0, -0.0, -1.54, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                LineArray::new(4, spacing).err(),
                Some(ArrayError::SpacingOutOfRange),
                "spacing {spacing}"
            );
        }
    }

    #[test]
    fn a_line_too_long_is_refused() {
        for (mics, spacing) in [(2, 1024.01), (64, 16.3), (4, 400.0), (2, f32::MAX)] {
            assert_eq!(
                LineArray::new(mics, spacing).err(),
                Some(ArrayError::TooLong),
                "{mics} microphones {spacing} apart"
            );
        }
    }
}
