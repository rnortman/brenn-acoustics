//! Signals and measures the unit tests share. Every signal is made in code
//! from a fixed seed.

use crate::array::LineArray;
use crate::fft::{self, Complex};
use std::f64::consts::PI;

/// A fixed-seed generator of uniform noise in [-1, 1).
pub(crate) struct Noise(pub(crate) u64);

impl Noise {
    pub(crate) fn next(&mut self) -> f32 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40;
        bits as f32 / (1u64 << 23) as f32 - 1.0
    }

    pub(crate) fn take(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.next()).collect()
    }
}

/// `signal` multiplied by the factor that gives it mean power `power`.
/// Panics on a silent signal.
fn scaled_to_power(mut signal: Vec<f32>, power: f64) -> Vec<f32> {
    let current = mean_square(&signal);
    assert!(current > 0.0, "a silent signal has no scale");
    let gain = (power / current).sqrt() as f32;
    for x in &mut signal {
        *x *= gain;
    }
    signal
}

/// Uniform white noise of mean power `power`.
pub(crate) fn white(len: usize, power: f64, seed: u64) -> Vec<f32> {
    scaled_to_power(Noise(seed).take(len), power)
}

/// Uniform white noise as long as `echo`, `below_db` under its mean power.
pub(crate) fn noise_below(echo: &[f32], below_db: f64, seed: u64) -> Vec<f32> {
    white(
        echo.len(),
        mean_square(echo) * 10f64.powf(-below_db / 10.0),
        seed,
    )
}

/// `signal` with every sample multiplied by `gain`.
pub(crate) fn amplified(signal: &[f32], gain: f32) -> Vec<f32> {
    signal.iter().map(|x| x * gain).collect()
}

/// The bit patterns of `signal`'s samples: two signals are bit-identical
/// when these are equal, which `==` on `f32` does not say of `-0.0` against
/// `0.0` or of NaN.
pub(crate) fn bits(signal: &[f32]) -> Vec<u32> {
    signal.iter().map(|x| x.to_bits()).collect()
}

/// The speech-like signal: uniform noise through the one-pole low-pass
/// `s = 0.9 s + w`, its envelope swung by `0.55 + 0.45 sin(2 pi 3 t / 16000)`,
/// scaled to mean power `power`.
pub(crate) fn low_passed(len: usize, power: f64, seed: u64) -> Vec<f32> {
    let mut noise = Noise(seed);
    let mut state = 0.0f32;
    let signal = (0..len)
        .map(|t| {
            state = 0.9 * state + noise.next();
            let swing = 0.55 + 0.45 * (2.0 * std::f64::consts::PI * 3.0 * t as f64 / 16000.0).sin();
            state * swing as f32
        })
        .collect();
    scaled_to_power(signal, power)
}

/// The test echo path: 1200 taps of uniform noise times `exp(-k / decay)`,
/// the first tap set to 5.2, the whole scaled to unit energy.
pub(crate) fn test_path(decay: f64, seed: u64) -> Vec<f32> {
    let mut noise = Noise(seed);
    let mut path: Vec<f64> = (0..1200)
        .map(|k| f64::from(noise.next()) * (-(k as f64) / decay).exp())
        .collect();
    path[0] = 5.2;
    let energy: f64 = path.iter().map(|h| h * h).sum();
    path.iter().map(|h| (h / energy.sqrt()) as f32).collect()
}

/// `signal` through `path`, causal, as long as `signal`: frames before the
/// start count as zero.
pub(crate) fn convolve(signal: &[f32], path: &[f32]) -> Vec<f32> {
    (0..signal.len())
        .map(|t| {
            path.iter()
                .take(t + 1)
                .enumerate()
                .map(|(k, h)| f64::from(*h) * f64::from(signal[t - k]))
                .sum::<f64>() as f32
        })
        .collect()
}

/// `a` with `b` added frame by frame. Panics on different lengths.
pub(crate) fn sum(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "summed signals differ in length");
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// The mean of the squared samples, in `f64`. Panics on an empty span.
pub(crate) fn mean_square(span: &[f32]) -> f64 {
    assert!(!span.is_empty(), "an empty span has no power");
    span.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / span.len() as f64
}

/// The mean power of `span` in dB. Panics on an empty span.
pub(crate) fn power_db(span: &[f32]) -> f64 {
    10.0 * mean_square(span).log10()
}

/// The gain of `s` in `out`, `sum(out s) / sum(s^2)`, in dB of amplitude.
/// Panics on different lengths, a silent `s` or a gain not above zero.
pub(crate) fn gain_db(out: &[f32], s: &[f32]) -> f64 {
    assert_eq!(out.len(), s.len(), "the two spans differ in length");
    let cross: f64 = out
        .iter()
        .zip(s)
        .map(|(o, x)| f64::from(*o) * f64::from(*x))
        .sum();
    let energy: f64 = s.iter().map(|x| f64::from(*x).powi(2)).sum();
    assert!(energy > 0.0, "a silent signal has no gain");
    let gain = cross / energy;
    assert!(gain > 0.0, "gain {gain} is not above zero");
    20.0 * gain.log10()
}

/// The amplitude of the sinusoid at `frequency` cycles per frame in
/// `signal`, fitted by least squares on a sine and a cosine from frame
/// `start` on.
pub(crate) fn amplitude(signal: &[f32], frequency: f64, start: usize) -> f64 {
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

/// How many frames before the array's centre microphone `m` hears a plane
/// wave from `angle`: `(m - (M - 1) / 2) * spacing * cos(angle)`.
fn advance(array: &LineArray, m: usize, angle: f32) -> f64 {
    let centre = (array.mics() - 1) as f64 / 2.0;
    (m as f64 - centre) * f64::from(array.spacing()) * f64::from(angle).cos()
}

/// One degree, in radians.
pub(crate) const DEGREE: f32 = std::f32::consts::PI / 180.0;

/// The array the multi-microphone tests share: four microphones 1.54 frames
/// apart, 33 mm at 16 kHz.
pub(crate) fn test_array() -> LineArray {
    LineArray::new(4, 1.54).unwrap()
}

/// `mics`, each with independent white noise of mean power `power` added,
/// microphone `m`'s from seed `seed + m`.
pub(crate) fn with_noise(mics: &[Vec<f32>], power: f64, seed: u64) -> Vec<Vec<f32>> {
    mics.iter()
        .enumerate()
        .map(|(m, c)| sum(c, &white(c.len(), power, seed + m as u64)))
        .collect()
}

/// Each microphone's copy of the unit sine at `frequency` cycles per frame
/// arriving from `angle`, `len` frames: microphone `m` is
/// `sin(2 pi frequency (t + advance))`, the array's centre hearing
/// `sin(2 pi frequency t)`.
pub(crate) fn plane_tone(
    array: &LineArray,
    angle: f32,
    frequency: f64,
    len: usize,
) -> Vec<Vec<f32>> {
    (0..array.mics())
        .map(|m| {
            let lead = advance(array, m, angle);
            (0..len)
                .map(|t| (2.0 * PI * frequency * (t as f64 + lead)).sin() as f32)
                .collect()
        })
        .collect()
}

/// A plane wave on a line array: what the array's centre hears, and each
/// microphone's copy.
pub(crate) struct PlaneWave {
    pub(crate) centre: Vec<f32>,
    pub(crate) mics: Vec<Vec<f32>>,
}

/// A random-phase source of flat spectrum arriving from `angle`, `len`
/// frames, a power of two: every bin from `band.0` to `band.1` cycles per
/// frame holds unit magnitude, the others zero, and microphone `m`'s copy is
/// the spectrum times the exact phase ramp of its advance, so that a delay
/// of any fraction of a frame is exact for the signal taken as periodic in
/// `len`. Bins 0 and `len / 2` are zero. Scaled so that the centre's mean
/// power is `power`. The same seed gives the same phases whatever the band.
pub(crate) fn plane_wave(
    array: &LineArray,
    angle: f32,
    band: (f64, f64),
    len: usize,
    power: f64,
    seed: u64,
) -> PlaneWave {
    shaped_plane_wave(
        array,
        angle,
        |f| (f >= band.0 && f <= band.1).then_some(1.0),
        len,
        power,
        seed,
    )
}

/// As [`plane_wave`] over the whole band, with the speech-like tilt of
/// [`low_passed`]: bin magnitudes follow the one-pole low-pass
/// `1 / |1 - 0.9 e^(-2 pi i f)|`, 25 dB down from the lowest frequencies to
/// half the rate.
pub(crate) fn tilted_plane_wave(
    array: &LineArray,
    angle: f32,
    len: usize,
    power: f64,
    seed: u64,
) -> PlaneWave {
    shaped_plane_wave(
        array,
        angle,
        |f| Some(1.0 / (1.81 - 1.8 * (2.0 * PI * f).cos()).sqrt()),
        len,
        power,
        seed,
    )
}

/// A random-phase source of `len` frames, a power of two, arriving from
/// `angle`, bin `k`'s magnitude being `magnitude(k / len)` and `None` an
/// empty bin. Bins 0 and `len / 2` are zero. Scaled so that the centre's
/// mean power is `power`.
fn shaped_plane_wave(
    array: &LineArray,
    angle: f32,
    magnitude: impl Fn(f64) -> Option<f64>,
    len: usize,
    power: f64,
    seed: u64,
) -> PlaneWave {
    let mut noise = Noise(seed);
    let bins: Vec<Option<(f64, f64)>> = (0..len / 2)
        .map(|k| {
            let phase = PI * f64::from(noise.next());
            let size = magnitude(k as f64 / len as f64);
            if k == 0 {
                None
            } else {
                size.map(|s| (s, phase))
            }
        })
        .collect();
    let delayed = |lead: f64| -> Vec<f32> {
        let mut spectrum = vec![Complex::ZERO; len];
        for (k, bin) in bins.iter().enumerate() {
            if let Some((size, phase)) = bin {
                let turn = phase + 2.0 * PI * k as f64 * lead / len as f64;
                let bin = Complex::new((size * turn.cos()) as f32, (size * turn.sin()) as f32);
                spectrum[k] = bin;
                spectrum[len - k] = bin.conj();
            }
        }
        fft::inverse(&mut spectrum);
        spectrum.iter().map(|x| x.re).collect()
    };
    let centre = delayed(0.0);
    let gain = (power / mean_square(&centre)).sqrt() as f32;
    PlaneWave {
        centre: amplified(&centre, gain),
        mics: (0..array.mics())
            .map(|m| amplified(&delayed(advance(array, m, angle)), gain))
            .collect(),
    }
}
