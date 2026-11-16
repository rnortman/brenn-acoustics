//! A radix-2 complex FFT and its inverse, in scalar `f32`.

use std::f32::consts::PI;

/// A complex number in rectangular form.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Complex {
    pub(crate) re: f32,
    pub(crate) im: f32,
}

impl Complex {
    pub(crate) const ZERO: Complex = Complex { re: 0.0, im: 0.0 };

    pub(crate) fn new(re: f32, im: f32) -> Complex {
        Complex { re, im }
    }

    pub(crate) fn conj(self) -> Complex {
        Complex::new(self.re, -self.im)
    }

    pub(crate) fn mul(self, other: Complex) -> Complex {
        Complex::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }

    pub(crate) fn add(self, other: Complex) -> Complex {
        Complex::new(self.re + other.re, self.im + other.im)
    }

    pub(crate) fn sub(self, other: Complex) -> Complex {
        Complex::new(self.re - other.re, self.im - other.im)
    }

    pub(crate) fn scale(self, factor: f32) -> Complex {
        Complex::new(self.re * factor, self.im * factor)
    }

    pub(crate) fn norm(self) -> f32 {
        self.re.hypot(self.im)
    }
}

/// The forward transform, in place: `X[k] = sum_n x[n] e^(-2 pi i k n / N)`.
///
/// Panics unless the length is a power of two.
pub(crate) fn forward(data: &mut [Complex]) {
    transform(data, -1.0);
}

/// The inverse transform, in place, scaled by `1 / N`, so that
/// `inverse(forward(x)) == x` up to rounding.
///
/// Panics unless the length is a power of two.
pub(crate) fn inverse(data: &mut [Complex]) {
    transform(data, 1.0);
    let scale = 1.0 / data.len() as f32;
    for value in data.iter_mut() {
        *value = value.scale(scale);
    }
}

/// Iterative decimation in time: bit-reversal permutation, then butterflies
/// of doubling span. `sign` is the sign of the exponent.
fn transform(data: &mut [Complex], sign: f32) {
    let n = data.len();
    assert!(n.is_power_of_two(), "FFT length {n} is not a power of two");
    if n == 1 {
        return;
    }
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if i < j {
            data.swap(i, j);
        }
    }
    let mut span = 2;
    while span <= n {
        let half = span / 2;
        let step = sign * 2.0 * PI / span as f32;
        for start in (0..n).step_by(span) {
            for k in 0..half {
                let (sin, cos) = (step * k as f32).sin_cos();
                let twiddle = Complex::new(cos, sin);
                let even = data[start + k];
                let odd = data[start + k + half].mul(twiddle);
                data[start + k] = even.add(odd);
                data[start + k + half] = even.sub(odd);
            }
        }
        span *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The transform computed from its definition, in `f64`.
    fn by_definition(input: &[Complex], sign: f64) -> Vec<(f64, f64)> {
        let n = input.len();
        (0..n)
            .map(|k| {
                input
                    .iter()
                    .enumerate()
                    .fold((0.0, 0.0), |(re, im), (t, x)| {
                        let angle = sign * 2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
                        let (s, c) = angle.sin_cos();
                        let (xr, xi) = (f64::from(x.re), f64::from(x.im));
                        (re + xr * c - xi * s, im + xr * s + xi * c)
                    })
            })
            .collect()
    }

    fn signal(n: usize) -> Vec<Complex> {
        (0..n)
            .map(|t| {
                let t = t as f32;
                Complex::new((0.37 * t).sin() + 0.25 * (1.3 * t).cos(), (0.11 * t).cos())
            })
            .collect()
    }

    #[test]
    fn forward_matches_the_definition() {
        for n in [1, 2, 4, 8, 16, 64, 256] {
            let input = signal(n);
            let mut data = input.clone();
            forward(&mut data);
            let expected = by_definition(&input, -1.0);
            let tolerance = 1e-4 * n as f64;
            for (k, (got, want)) in data.iter().zip(&expected).enumerate() {
                assert!(
                    (f64::from(got.re) - want.0).abs() < tolerance
                        && (f64::from(got.im) - want.1).abs() < tolerance,
                    "n={n} k={k}: got {got:?}, want {want:?}"
                );
            }
        }
    }

    #[test]
    fn inverse_matches_the_definition() {
        let n = 32;
        let input = signal(n);
        let mut data = input.clone();
        inverse(&mut data);
        let expected = by_definition(&input, 1.0);
        for (k, (got, want)) in data.iter().zip(&expected).enumerate() {
            let (want_re, want_im) = (want.0 / n as f64, want.1 / n as f64);
            assert!(
                (f64::from(got.re) - want_re).abs() < 1e-5
                    && (f64::from(got.im) - want_im).abs() < 1e-5,
                "k={k}: got {got:?}, want ({want_re}, {want_im})"
            );
        }
    }

    #[test]
    fn inverse_undoes_forward() {
        for n in [1, 2, 8, 128, 1024] {
            let input = signal(n);
            let mut data = input.clone();
            forward(&mut data);
            inverse(&mut data);
            for (t, (got, want)) in data.iter().zip(&input).enumerate() {
                assert!(
                    (got.re - want.re).abs() < 1e-4 && (got.im - want.im).abs() < 1e-4,
                    "n={n} t={t}: got {got:?}, want {want:?}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "not a power of two")]
    fn refuses_a_length_that_is_not_a_power_of_two() {
        forward(&mut [Complex::ZERO; 6]);
    }
}
