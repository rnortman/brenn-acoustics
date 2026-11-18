//! Acoustics algorithms for microphone arrays, free of I/O and of any
//! hardware dependency.
//!
//! Every function here takes samples and returns results: nothing reads or
//! writes a file, a device or the network. Samples are plain scalar `f32`.
//!
//! - [`delay`]: the delay between two channels, to a fraction of a frame
//!   ([`delay::gcc_phat`]), and where the echo of a reference sits in a
//!   microphone, in whole frames ([`delay::reference_delay`]).
//! - [`echo`]: a per-microphone echo canceller, fed in blocks
//!   ([`echo::Canceller`]).
//! - [`filter`]: a high-pass to put ahead of it ([`filter::HighPass`]).
//!
//! The three compose as follows. High-pass the reference and each
//! microphone, one [`filter::HighPass`] each. Find the delay once, with
//! [`delay::reference_delay`] over a span in which the reference sounds.
//! Pass its `frames`, or 0 when it is negative, as the `echo_delay` of one
//! [`echo::Canceller`] per microphone; an echo leading its reference by no
//! more than the canceller's `lead` is covered by the lead taps. Lay each
//! output over its microphone by [`echo::Canceller::output_lag`].
//!
//! For a line of microphones:
//!
//! - [`array`]: the line's geometry ([`array::LineArray`]) and the
//!   conventions every multi-microphone algorithm shares: how channels are
//!   passed and how a direction is measured.
//! - [`beam`]: delay-and-sum beams at fixed angles, fed in blocks
//!   ([`beam::FixedBeams`]), and one channel that moves to the beam nearest
//!   the dominant source's direction ([`beam::SelectedBeam`]).
//! - [`doa`]: the direction of the dominant source, read at a fixed
//!   interval ([`doa::Direction`]).

pub mod array;
pub mod beam;
pub mod delay;
pub mod doa;
pub mod echo;
mod fft;
pub mod filter;
#[cfg(test)]
mod test_signals;
