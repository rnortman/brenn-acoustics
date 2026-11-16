//! Signals and measures the unit tests share. Every signal is made in code
//! from a fixed seed.

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
