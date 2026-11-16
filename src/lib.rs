//! Acoustics algorithms for microphone arrays, free of I/O and of any
//! hardware dependency.
//!
//! Every function here takes samples and returns results: nothing reads or
//! writes a file, a device or the network. Samples are plain scalar `f32`.
//!
//! - [`delay`]: the delay between two channels, to a fraction of a frame.

pub mod delay;
mod fft;
