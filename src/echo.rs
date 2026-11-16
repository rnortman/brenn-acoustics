//! A per-microphone adaptive echo canceller, fed in blocks.
//!
//! [`Canceller`] takes the loudspeaker's reference and one microphone and
//! returns the microphone with its estimate of the reference's echo taken
//! out. Four microphones are four cancellers fed the same reference.

use std::fmt;

/// While the error's smoothed power is above this many times the estimate's,
/// the step is slowed.
const SLOW_RATIO: f32 = 4.0;
/// The share of the step taken while it is slowed.
const SLOW_SHARE: f32 = 0.125;

/// How a [`Canceller`] is built. There is no default; the values the
/// crate's tests hold it at, for 16 kHz and samples of full scale 1, are
/// `taps` 1024, `lead` 32, `step` 0.1 and `floor` 1e-6, with `echo_delay`
/// as [`crate::delay::reference_delay`] measures it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CancellerConfig {
    /// The filter's length, at least 1.
    pub taps: usize,
    /// Taps placed ahead of `echo_delay`, below `taps`. The output lags the
    /// microphone by this many frames.
    pub lead: usize,
    /// The echo's delay behind the reference, in whole frames: what
    /// [`crate::delay::reference_delay`] measures. Not the output's lag.
    pub echo_delay: usize,
    /// The adaptation step, above 0 and at most 1.
    pub step: f32,
    /// The reference power per frame, in squared sample units, at which the
    /// normalisation stops following the reference down. Above 0, with
    /// `taps * floor` a finite, normal (not subnormal) `f32`.
    pub floor: f32,
}

/// Why a [`Canceller`] refused its configuration or its input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EchoError {
    /// `taps` is 0.
    NoTaps,
    /// `lead` is not below `taps`.
    LeadTooLarge { lead: usize, taps: usize },
    /// `step` is not within `(0, 1]`, or not finite.
    StepOutOfRange,
    /// `floor` is not above 0, or `taps * floor` is not a finite, normal
    /// `f32`.
    FloorOutOfRange,
    /// The `echo_delay + taps` frames of reference history, twice over,
    /// cannot be counted in a `usize` or allocated.
    HistoryTooLong { echo_delay: usize, taps: usize },
    /// The reference and microphone blocks differ in length.
    LengthMismatch { reference: usize, mic: usize },
    /// A sample in either block is NaN or infinite.
    NonFinite,
}

impl fmt::Display for EchoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EchoError::NoTaps => write!(f, "the filter needs at least one tap"),
            EchoError::LeadTooLarge { lead, taps } => {
                write!(f, "a lead of {lead} taps is not below the {taps} taps")
            }
            EchoError::StepOutOfRange => write!(f, "the step is not above 0 and at most 1"),
            EchoError::FloorOutOfRange => write!(
                f,
                "the floor is not above 0 with taps times floor a finite, normal f32"
            ),
            EchoError::HistoryTooLong { echo_delay, taps } => write!(
                f,
                "a history of {echo_delay} + {taps} frames, twice over, cannot be allocated"
            ),
            EchoError::LengthMismatch { reference, mic } => write!(
                f,
                "block lengths differ: reference {reference}, microphone {mic} frames"
            ),
            EchoError::NonFinite => write!(f, "a sample is not finite"),
        }
    }
}

impl std::error::Error for EchoError {}

/// A streaming echo canceller for one microphone: normalised least mean
/// squares in the time domain, with regularisation.
///
/// With `x` the reference, `d` the microphone and `n` counting frames from
/// the first one ever fed, each frame computes
///
/// ```text
/// u[k]   = x[n - echo_delay - k]           k in 0..taps, zero before the start
/// est    = sum of w[k] u[k]
/// e      = d[n - lead] - est               d zero before the start
/// out[n] = e
/// Pe    += (e^2 - Pe) a ;  Py += (est^2 - Py) a          a = min(1, 4 / taps)
/// mu     = step  if Pe <= 4 Py  else  step / 8
/// w[k]  += mu e u[k] / (sum of u[k]^2 + taps floor)
/// ```
///
/// with every state variable starting at zero. Tap `k` stands for a path
/// delay of `echo_delay - lead + k` frames: the filter covers the echo from
/// `lead` frames before `echo_delay` to `taps - lead` after it. An echo that
/// leads its reference by no more than `lead` frames, as on a reference
/// placed on a recording afterwards, is covered with `echo_delay` 0.
///
/// **The output lag.** `out[n]` is the microphone's frame `n - lead` with
/// the estimate taken out: the first `lead` output frames are zero, and the
/// microphone's last `lead` frames come out only once `lead` more frames are
/// fed. [`Canceller::output_lag`] is that lag, whatever `echo_delay` is.
///
/// **Adaptation.** The step is slowed to an eighth of `step` while the
/// error's smoothed power is above 4 times the estimate's, over a smoothing
/// span of a quarter of the filter, and is the full step otherwise. Nothing
/// ever stops adaptation, so the filter starts from zero at the slow step
/// and re-adapts after any change of the path.
///
/// **A near-end talker.** While the near end is about as loud as the echo
/// the condition above is rarely met and the filter adapts at the full step
/// on the talker too. Each update stays in the estimate on later frames to
/// the degree the reference resembles itself across that lag, and takes out
/// what of the near end resembles itself across the same lag: the near
/// end's gain at the output is about `1 / (1 + step sum(rx(k) rs(k)))`, the
/// sum over lags `k >= 1` of the product of the reference's and the near
/// end's normalised autocorrelations. With a white reference or a white near
/// end that is 0 dB; with a near end and a reference of one low-passed
/// spectrum at `step` 0.1 it is about -3 dB. A near end well above the echo
/// slows the step and keeps what the filter learned.
///
/// **Silence and scale.** A reference of zeros leaves the filter at zero and
/// the output the delayed microphone exactly. `taps * floor` bounds the
/// update while the reference is small and not zero; `floor` is a level in
/// the caller's units. The canceller works in the caller's units and is
/// meant for samples of full scale about 1: a sample near `f32::MAX` can
/// overflow inside it, and the output is then not finite.
///
/// **Memory.** The canceller keeps `echo_delay + taps` frames of reference
/// history, twice over; there is no cap. A history the allocator refuses is
/// an error, but an allocator that overcommits memory can accept one it
/// cannot later supply.
#[derive(Clone, Debug)]
pub struct Canceller {
    config: CancellerConfig,
    /// The coefficients, last tap first: `weights[j]` is `w[taps - 1 - j]`.
    weights: Vec<f32>,
    /// The reference's last `span` frames, stored twice over so that they
    /// are always one contiguous slice: frame `n` sits at `n % span` and at
    /// `n % span + span`.
    history: Vec<f32>,
    /// `echo_delay + taps`.
    span: usize,
    /// The microphone's last `lead` frames: frame `n` sits at `n % lead`.
    mic_line: Vec<f32>,
    /// Frames fed so far.
    frame: u64,
    /// The running `sum of u[k]^2`.
    power: f32,
    /// `Pe`.
    error_power: f32,
    /// `Py`.
    estimate_power: f32,
    /// `a`.
    smoothing: f32,
    /// `taps * floor`.
    regulariser: f32,
}

impl Canceller {
    /// A canceller with every coefficient zero. Refuses `taps` 0, `lead` not
    /// below `taps`, a `step` not within `(0, 1]`, a `floor` not above 0 or
    /// whose `taps * floor` is not a finite, normal `f32`, and an
    /// `echo_delay + taps` frames of history, twice over, that cannot be
    /// counted in a `usize` or that the allocator refuses.
    pub fn new(config: CancellerConfig) -> Result<Canceller, EchoError> {
        if config.taps == 0 {
            return Err(EchoError::NoTaps);
        }
        if config.lead >= config.taps {
            return Err(EchoError::LeadTooLarge {
                lead: config.lead,
                taps: config.taps,
            });
        }
        if !(config.step > 0.0 && config.step <= 1.0) {
            return Err(EchoError::StepOutOfRange);
        }
        // The regulariser alone divides the update while the reference is
        // silent: a subnormal one can overflow the gain, an infinite one
        // stops adaptation.
        let regulariser = config.taps as f32 * config.floor;
        if !(config.floor > 0.0 && regulariser.is_normal()) {
            return Err(EchoError::FloorOutOfRange);
        }
        let too_long = EchoError::HistoryTooLong {
            echo_delay: config.echo_delay,
            taps: config.taps,
        };
        let span = config.taps.checked_add(config.echo_delay).ok_or(too_long)?;
        // `taps` and `lead` are below `span`, so the history is the largest
        // of the three buffers.
        let history = span.checked_mul(2).and_then(zeros).ok_or(too_long)?;
        Ok(Canceller {
            config,
            weights: zeros(config.taps).ok_or(too_long)?,
            history,
            span,
            mic_line: zeros(config.lead).ok_or(too_long)?,
            frame: 0,
            power: 0.0,
            error_power: 0.0,
            estimate_power: 0.0,
            smoothing: (4.0 / config.taps as f32).min(1.0),
            regulariser,
        })
    }

    /// The frames by which the output lags the microphone: `lead`, whatever
    /// `echo_delay` is.
    pub fn output_lag(&self) -> usize {
        self.config.lead
    }

    /// Feeds one block, continuing from the previous call; `mic` is replaced
    /// by the output. Any cut of the input into blocks gives bit-identical
    /// output, and a block of length zero does nothing. Blocks of different
    /// lengths and a sample that is not finite in either are refused before
    /// anything changes, leaving the canceller and `mic` as they were.
    pub fn process(&mut self, reference: &[f32], mic: &mut [f32]) -> Result<(), EchoError> {
        if reference.len() != mic.len() {
            return Err(EchoError::LengthMismatch {
                reference: reference.len(),
                mic: mic.len(),
            });
        }
        if !reference.iter().chain(mic.iter()).all(|x| x.is_finite()) {
            return Err(EchoError::NonFinite);
        }
        for (x, d) in reference.iter().zip(mic.iter_mut()) {
            *d = self.advance(*x, *d);
        }
        Ok(())
    }

    /// One frame of the recurrence: takes the reference's and the
    /// microphone's frame `n` and returns `out[n]`.
    fn advance(&mut self, x: f32, d: f32) -> f32 {
        let taps = self.config.taps;
        let span = self.span;
        let at = (self.frame % span as u64) as usize;
        let leaving = self.history[at];
        self.history[at] = x;
        self.history[at + span] = x;
        // `window[j]` is `u[taps - 1 - j]`.
        let window = &self.history[at + 1..at + 1 + taps];

        // The running sum is recomputed every `taps` frames so that rounding
        // neither accumulates nor depends on how the input was cut.
        self.power = if self.frame.is_multiple_of(taps as u64) {
            window.iter().map(|u| u * u).sum()
        } else {
            let entering = window[taps - 1];
            (self.power + entering * entering - leaving * leaving).max(0.0)
        };

        // TODO(loudspeaker-nonlinearity-model): the path is modelled as linear.
        let estimate: f32 = self.weights.iter().zip(window).map(|(w, u)| w * u).sum();

        let lead = self.config.lead;
        let delayed = if lead == 0 {
            d
        } else {
            let slot = (self.frame % lead as u64) as usize;
            std::mem::replace(&mut self.mic_line[slot], d)
        };
        let error = delayed - estimate;

        self.error_power += (error * error - self.error_power) * self.smoothing;
        self.estimate_power += (estimate * estimate - self.estimate_power) * self.smoothing;
        let step = if self.error_power <= SLOW_RATIO * self.estimate_power {
            self.config.step
        } else {
            self.config.step * SLOW_SHARE
        };
        let gain = step * error / (self.power + self.regulariser);
        for (w, u) in self.weights.iter_mut().zip(window) {
            *w += gain * u;
        }

        self.frame += 1;
        error
    }
}

/// `len` zeros, or `None` when the allocator refuses them.
fn zeros(len: usize) -> Option<Vec<f32>> {
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(len).ok()?;
    buffer.resize(len, 0.0);
    Some(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delay::reference_delay;
    use crate::filter::HighPass;
    use crate::test_signals::{
        Noise, amplified, bits, convolve, gain_db, low_passed, mean_square, noise_below, power_db,
        sum, test_path, white,
    };

    const FS: usize = 16000;
    const TAPS: usize = 1024;
    const LEAD: usize = 32;
    const BULK: usize = 40;
    /// The power of uniform noise of amplitude 0.1.
    const REFERENCE_POWER: f64 = 0.01 / 3.0;

    fn config(echo_delay: usize) -> CancellerConfig {
        CancellerConfig {
            taps: TAPS,
            lead: LEAD,
            echo_delay,
            step: 0.1,
            floor: 1e-6,
        }
    }

    fn frames(seconds: f64) -> usize {
        (seconds * FS as f64).round() as usize
    }

    /// A scene of `seconds`, with `lead` frames more so that the output for
    /// the last microphone frame exists.
    fn scene_len(seconds: f64) -> usize {
        frames(seconds) + LEAD
    }

    /// The test path behind the bulk delay.
    fn room(decay: f64, seed: u64) -> Vec<f32> {
        let mut path = vec![0.0; BULK];
        path.extend(test_path(decay, seed));
        path
    }

    /// Zeros of length `len` with independent bursts from `make` (length,
    /// power, seed) over each of `bursts`, in seconds.
    fn bursts(
        len: usize,
        bursts: &[(f64, f64)],
        power: f64,
        seed: u64,
        make: fn(usize, f64, u64) -> Vec<f32>,
    ) -> Vec<f32> {
        let mut signal = vec![0.0; len];
        for (i, &(start, end)) in bursts.iter().enumerate() {
            let (a, b) = (frames(start), frames(end));
            signal[a..b].copy_from_slice(&make(b - a, power, seed + i as u64));
        }
        signal
    }

    /// The output of a fresh canceller over the whole of `mic`, and the
    /// canceller.
    fn cancel(echo_delay: usize, reference: &[f32], mic: &[f32]) -> (Canceller, Vec<f32>) {
        let mut canceller = Canceller::new(config(echo_delay)).unwrap();
        let mut out = mic.to_vec();
        canceller.process(reference, &mut out).unwrap();
        (canceller, out)
    }

    fn span(signal: &[f32], start: f64, end: f64) -> &[f32] {
        &signal[frames(start)..frames(end)]
    }

    /// The output frames that carry the microphone's frames over
    /// `[start, end)` seconds.
    fn out_span(out: &[f32], start: f64, end: f64) -> &[f32] {
        &out[frames(start) + LEAD..frames(end) + LEAD]
    }

    /// The output's power over the noise's on the same microphone frames, dB.
    fn over_noise(out: &[f32], noise: &[f32], start: f64, end: f64) -> f64 {
        power_db(out_span(out, start, end)) - power_db(span(noise, start, end))
    }

    /// The microphone's power over the output's on the same frames, dB.
    fn reduction(mic: &[f32], out: &[f32], start: f64, end: f64) -> f64 {
        power_db(span(mic, start, end)) - power_db(out_span(out, start, end))
    }

    /// The output's power over the summed powers of `near` and `noise`, dB.
    fn over_near_and_noise(out: &[f32], near: &[f32], noise: &[f32], start: f64, end: f64) -> f64 {
        let expected = mean_square(span(near, start, end)) + mean_square(span(noise, start, end));
        power_db(out_span(out, start, end)) - 10.0 * expected.log10()
    }

    /// The near end's gain in the output over `[start, end)`, the lead
    /// allowed for.
    fn near_gain(out: &[f32], near: &[f32], start: f64, end: f64) -> f64 {
        gain_db(out_span(out, start, end), span(near, start, end))
    }

    /// The microphone frame `n - lead`, zero before the start.
    fn delayed(mic: &[f32], lead: usize) -> Vec<f32> {
        let mut out = vec![0.0; mic.len()];
        out[lead..].copy_from_slice(&mic[..mic.len() - lead]);
        out
    }

    /// The first test's scene: a white reference through the room, noise 20
    /// dB below the echo. Returns the reference, the microphone and the
    /// noise.
    fn white_scene(seconds: f64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let reference = white(scene_len(seconds), REFERENCE_POWER, 1);
        let echo = convolve(&reference, &room(200.0, 2));
        let noise = noise_below(&echo, 20.0, 3);
        let mic = sum(&echo, &noise);
        (reference, mic, noise)
    }

    #[test]
    fn aligned_white_reference_converges() {
        let (reference, mic, noise) = white_scene(6.0);
        let delay = reference_delay(&reference[..FS], &mic[..FS], 2000).unwrap();
        assert_eq!(delay.frames, BULK as isize);
        let (_, out) = cancel(delay.frames as usize, &reference, &mic);
        let early = over_noise(&out, &noise, 0.0, 1.0);
        assert!(early >= 15.0, "over [0, 1] {early} dB over the noise");
        // Converged within 5 s.
        let late = over_noise(&out, &noise, 5.0, 6.0);
        assert!(late <= 1.5, "over [5, 6] {late} dB over the noise");
    }

    /// Over [5, 6], the output's power over the noise's, dB, for a white
    /// reference whose echo leads it by `lead_by` frames, cancelled with
    /// `echo_delay` 0.
    fn leading_echo_over_noise(lead_by: usize, seed: u64) -> f64 {
        let len = scene_len(6.0);
        let source = white(len + lead_by, REFERENCE_POWER, seed);
        let echo = convolve(&source, &test_path(200.0, seed + 1))[lead_by..].to_vec();
        let reference = source[..len].to_vec();
        let noise = noise_below(&echo, 20.0, seed + 2);
        let (_, out) = cancel(0, &reference, &sum(&echo, &noise));
        over_noise(&out, &noise, 5.0, 6.0)
    }

    /// At exactly `LEAD` the echo's direct path sits on tap 0: a mapping one
    /// frame short of it leaves the output about 17 dB over the noise.
    #[test]
    fn echo_leading_by_up_to_the_lead_is_cancelled() {
        for lead_by in [LEAD / 2, LEAD] {
            let late = leading_echo_over_noise(lead_by, 24);
            assert!(
                late <= 1.5,
                "leading by {lead_by}: over [5, 6] {late} dB over the noise"
            );
        }
    }

    /// Over [5, 6], the output's power over the high-passed noise's, dB, for
    /// the white scene with a 30 Hz sinusoid 10 dB above the echo in the
    /// microphone only, put through the crate's recipe: both channels
    /// high-passed at 200 Hz, the delay found over the first second, then
    /// cancelled.
    fn composed_over_noise(seed: u64) -> f64 {
        let len = scene_len(6.0);
        let reference = white(len, REFERENCE_POWER, seed);
        let echo = convolve(&reference, &room(200.0, seed + 1));
        let noise = noise_below(&echo, 20.0, seed + 2);
        let amplitude = (20.0 * mean_square(&echo)).sqrt();
        let hum: Vec<f32> = (0..len)
            .map(|t| {
                let phase = 2.0 * std::f64::consts::PI * 30.0 * t as f64 / FS as f64;
                (amplitude * phase.sin()) as f32
            })
            .collect();
        let mic = sum(&sum(&echo, &noise), &hum);
        let high_passed = |signal: &[f32]| {
            let mut out = signal.to_vec();
            HighPass::new(200.0 / FS as f32)
                .unwrap()
                .process(&mut out)
                .unwrap();
            out
        };
        let (reference, mic, noise) = (
            high_passed(&reference),
            high_passed(&mic),
            high_passed(&noise),
        );
        let delay = reference_delay(&reference[..FS], &mic[..FS], 2000).unwrap();
        assert_eq!(delay.frames, BULK as isize, "{delay:?}");
        let (_, out) = cancel(delay.frames as usize, &reference, &mic);
        over_noise(&out, &noise, 5.0, 6.0)
    }

    #[test]
    fn high_pass_alignment_and_canceller_compose() {
        let late = composed_over_noise(25);
        assert!(
            late <= 2.0,
            "over [5, 6] {late} dB over the high-passed noise"
        );
    }

    #[test]
    fn speech_like_reference_converges() {
        let reference = low_passed(scene_len(10.0), REFERENCE_POWER, 4);
        let echo = convolve(&reference, &room(200.0, 2));
        let noise = noise_below(&echo, 15.0, 5);
        let (_, out) = cancel(BULK, &reference, &sum(&echo, &noise));
        let late = over_noise(&out, &noise, 9.0, 10.0);
        assert!(late <= 3.0, "over [9, 10] {late} dB over the noise");
    }

    #[test]
    fn white_near_end_as_loud_as_the_echo_passes() {
        let len = scene_len(13.0);
        let reference = white(len, REFERENCE_POWER, 6);
        let echo = convolve(&reference, &room(200.0, 2));
        let noise = noise_below(&echo, 20.0, 7);
        let near = bursts(
            len,
            &[(6.0, 7.5), (9.0, 11.0)],
            mean_square(&echo),
            8,
            white,
        );
        let mic = sum(&sum(&echo, &noise), &near);
        let (_, out) = cancel(BULK, &reference, &mic);
        for (start, end) in [(6.0, 7.5), (9.0, 11.0)] {
            let gain = near_gain(&out, &near, start, end);
            assert!(
                gain.abs() <= 0.5,
                "near end's gain {gain} dB over [{start}, {end}]"
            );
            let level = over_near_and_noise(&out, &near, &noise, start, end);
            assert!(
                level.abs() <= 1.0,
                "output {level} dB over near end and noise"
            );
        }
        let before = reduction(&mic, &out, 5.0, 6.0);
        for (start, end) in [(8.0, 9.0), (12.0, 13.0)] {
            let after = reduction(&mic, &out, start, end);
            assert!(
                (after - before).abs() <= 3.0,
                "reduction {after} dB over [{start}, {end}] against {before} dB before"
            );
        }
    }

    #[test]
    fn speech_like_near_end_on_speech_like_reference_loses_the_documented_gain() {
        let len = scene_len(13.0);
        let reference = low_passed(len, REFERENCE_POWER, 9);
        let echo = convolve(&reference, &room(200.0, 2));
        let noise = noise_below(&echo, 20.0, 10);
        let near = bursts(
            len,
            &[(6.0, 7.5), (9.0, 11.0)],
            mean_square(&echo),
            11,
            low_passed,
        );
        let mic = sum(&sum(&echo, &noise), &near);
        let (_, out) = cancel(BULK, &reference, &mic);
        for (start, end) in [(6.0, 7.5), (9.0, 11.0)] {
            let gain = near_gain(&out, &near, start, end);
            assert!(
                (-5.0..=-1.5).contains(&gain),
                "near end's gain {gain} dB over [{start}, {end}]"
            );
        }
    }

    #[test]
    fn near_end_well_above_the_echo_slows_adaptation() {
        let len = scene_len(9.0);
        let reference = white(len, REFERENCE_POWER, 12);
        let echo = convolve(&reference, &room(200.0, 2));
        let noise = noise_below(&echo, 20.0, 13);
        let near = bursts(len, &[(6.0, 7.5)], 10.0 * mean_square(&echo), 14, white);
        let mic = sum(&sum(&echo, &noise), &near);
        let (_, out) = cancel(BULK, &reference, &mic);
        let gain = near_gain(&out, &near, 6.0, 7.5);
        assert!(gain.abs() <= 0.5, "near end's gain {gain} dB");
        let level = over_near_and_noise(&out, &near, &noise, 6.0, 7.5);
        assert!(
            level.abs() <= 0.5,
            "output {level} dB over near end and noise"
        );
        let before = reduction(&mic, &out, 5.0, 6.0);
        let after = reduction(&mic, &out, 7.5, 9.0);
        assert!(
            (after - before).abs() <= 5.0,
            "reduction {after} dB after the burst against {before} dB before"
        );
    }

    /// The echo of a white reference through one room until `change`
    /// seconds and through `second` times `scale` from then on, with noise
    /// 20 dB below the first echo. Returns the reference, the microphone and
    /// the noise.
    fn changed_path(seconds: f64, change: f64, scale: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let reference = white(scene_len(seconds), REFERENCE_POWER, 15);
        let first = convolve(&reference, &room(200.0, 2));
        let second = convolve(&reference, &room(150.0, 16));
        let at = frames(change);
        let echo: Vec<f32> = first[..at]
            .iter()
            .copied()
            .chain(second[at..].iter().map(|x| x * scale))
            .collect();
        let noise = noise_below(&first, 20.0, 17);
        let mic = sum(&echo, &noise);
        (reference, mic, noise)
    }

    #[test]
    fn replaced_path_is_followed() {
        let (reference, mic, noise) = changed_path(12.0, 7.0, 1.0);
        let (_, out) = cancel(BULK, &reference, &mic);
        let after = over_noise(&out, &noise, 7.05, 7.5);
        assert!(after >= 10.0, "over [7.05, 7.5] {after} dB over the noise");
        // Back within 4 s.
        let late = over_noise(&out, &noise, 11.0, 12.0);
        assert!(late <= 1.5, "over [11, 12] {late} dB over the noise");
    }

    #[test]
    fn louder_replaced_path_is_followed() {
        let (reference, mic, noise) = changed_path(14.0, 7.0, 10f32.powf(12.0 / 20.0));
        let (_, out) = cancel(BULK, &reference, &mic);
        // Back within 6 s.
        let late = over_noise(&out, &noise, 13.0, 14.0);
        assert!(late <= 1.5, "over [13, 14] {late} dB over the noise");
    }

    #[test]
    fn silent_reference_passes_the_microphone_delayed() {
        let mic = white(frames(2.0), 1e-3, 18);
        let (canceller, out) = cancel(BULK, &vec![0.0; mic.len()], &mic);
        assert_eq!(bits(&out), bits(&delayed(&mic, LEAD)));
        assert!(canceller.weights.iter().all(|w| *w == 0.0));
    }

    #[test]
    fn reference_that_starts_silent() {
        let len = scene_len(8.0);
        let mut reference = white(len, REFERENCE_POWER, 19);
        let silence = frames(2.0);
        reference[..silence].fill(0.0);
        let echo = convolve(&reference, &room(200.0, 2));
        let noise = white(len, mean_square(&echo[silence..]) / 100.0, 20);
        let mic = sum(&echo, &noise);
        let (canceller, out) = cancel(BULK, &reference, &mic);
        assert_eq!(bits(&out[..silence]), bits(&delayed(&mic, LEAD)[..silence]));
        let late = over_noise(&out, &noise, 7.0, 8.0);
        assert!(late <= 1.5, "over [7, 8] {late} dB over the noise");
        let largest = canceller.weights.iter().fold(0.0f32, |m, w| m.max(w.abs()));
        assert!(largest <= 2.0, "a coefficient of magnitude {largest}");
    }

    #[test]
    fn near_silent_reference_keeps_the_filter_small() {
        let len = frames(3.0);
        let reference = amplified(&Noise(21).take(len), 1e-5);
        let mic = amplified(&Noise(22).take(len), 1e-3);
        let (canceller, out) = cancel(BULK, &reference, &mic);
        let largest = canceller.weights.iter().fold(0.0f32, |m, w| m.max(w.abs()));
        assert!(largest <= 1e-3, "a coefficient of magnitude {largest}");
        let level = power_db(&out[LEAD..]) - power_db(&mic[..len - LEAD]);
        assert!(level.abs() <= 0.01, "output {level} dB over the microphone");
    }

    #[test]
    fn any_cut_into_blocks_gives_the_same_output() {
        let (reference, mic, _) = white_scene(2.0);
        let (_, whole) = cancel(BULK, &reference, &mic);
        let irregular = [0, 1, 1023, 0, 5000, 17, 0, 160, 2048, 3];
        let cuts: [&[usize]; 5] = [&[1], &[7], &[160], &[1024], &irregular];
        for sizes in cuts {
            let mut canceller = Canceller::new(config(BULK)).unwrap();
            let mut out = mic.clone();
            let mut start = 0;
            for &size in sizes.iter().cycle() {
                if start == out.len() {
                    break;
                }
                let end = (start + size).min(out.len());
                canceller
                    .process(&reference[start..end], &mut out[start..end])
                    .unwrap();
                start = end;
            }
            assert_eq!(
                bits(&out),
                bits(&whole),
                "blocks of {sizes:?} differ from the whole"
            );
        }
    }

    #[test]
    fn output_lag_is_the_lead() {
        for lead in [0, LEAD] {
            for echo_delay in [0, BULK] {
                let canceller = Canceller::new(CancellerConfig {
                    lead,
                    ..config(echo_delay)
                })
                .unwrap();
                assert_eq!(canceller.output_lag(), lead);
            }
        }
        let mic = white(1000, 1e-3, 23);
        let mut canceller = Canceller::new(CancellerConfig {
            lead: 0,
            ..config(BULK)
        })
        .unwrap();
        let mut out = mic.clone();
        canceller.process(&vec![0.0; mic.len()], &mut out).unwrap();
        assert_eq!(bits(&out), bits(&mic));
    }

    #[test]
    fn invalid_configurations_are_refused() {
        let refused = |change: fn(&mut CancellerConfig), error: EchoError| {
            let mut config = config(BULK);
            change(&mut config);
            assert_eq!(Canceller::new(config).unwrap_err(), error, "{config:?}");
        };
        refused(|c| c.taps = 0, EchoError::NoTaps);
        refused(
            |c| c.lead = TAPS,
            EchoError::LeadTooLarge {
                lead: TAPS,
                taps: TAPS,
            },
        );
        refused(
            |c| c.lead = TAPS + 1,
            EchoError::LeadTooLarge {
                lead: TAPS + 1,
                taps: TAPS,
            },
        );
        refused(|c| c.step = 0.0, EchoError::StepOutOfRange);
        refused(|c| c.step = -0.1, EchoError::StepOutOfRange);
        refused(|c| c.step = 1.5, EchoError::StepOutOfRange);
        refused(|c| c.step = f32::NAN, EchoError::StepOutOfRange);
        refused(|c| c.floor = 0.0, EchoError::FloorOutOfRange);
        refused(|c| c.floor = -1e-6, EchoError::FloorOutOfRange);
        refused(|c| c.floor = f32::NAN, EchoError::FloorOutOfRange);
        refused(|c| c.floor = f32::INFINITY, EchoError::FloorOutOfRange);
        // `taps * floor` subnormal, and overflowing to infinity.
        refused(|c| c.floor = 1e-42, EchoError::FloorOutOfRange);
        refused(|c| c.floor = 1e36, EchoError::FloorOutOfRange);
        let smallest = CancellerConfig {
            taps: 1,
            lead: 0,
            floor: f32::MIN_POSITIVE,
            ..config(BULK)
        };
        assert!(Canceller::new(smallest).is_ok(), "{smallest:?}");
        // A negative delay cast to `usize`, a span whose double overflows,
        // and a history beyond what any allocation can hold.
        for echo_delay in [(-3isize) as usize, usize::MAX / 2, usize::MAX / 8] {
            let config = config(echo_delay);
            assert_eq!(
                Canceller::new(config).unwrap_err(),
                EchoError::HistoryTooLong {
                    echo_delay,
                    taps: TAPS
                },
                "{config:?}"
            );
        }
    }

    #[test]
    fn invalid_blocks_are_refused_and_change_nothing() {
        let (reference, mic, _) = white_scene(0.5);
        let half = reference.len() / 2;
        let mut untouched = Canceller::new(config(BULK)).unwrap();
        let mut expected = mic.clone();
        untouched.process(&reference, &mut expected).unwrap();

        let mut canceller = Canceller::new(config(BULK)).unwrap();
        let mut out = mic.clone();
        canceller
            .process(&reference[..half], &mut out[..half])
            .unwrap();

        let mut short = mic[half..half + 9].to_vec();
        assert_eq!(
            canceller.process(&reference[half..half + 10], &mut short),
            Err(EchoError::LengthMismatch {
                reference: 10,
                mic: 9
            })
        );
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for in_reference in [true, false] {
                let mut bad_reference = reference[half..half + 10].to_vec();
                let mut bad_mic = mic[half..half + 10].to_vec();
                if in_reference {
                    bad_reference[3] = bad;
                } else {
                    bad_mic[3] = bad;
                }
                let before = bad_mic.clone();
                assert_eq!(
                    canceller.process(&bad_reference, &mut bad_mic),
                    Err(EchoError::NonFinite)
                );
                assert_eq!(bits(&bad_mic), bits(&before), "a refused block was written");
            }
        }

        canceller
            .process(&reference[half..], &mut out[half..])
            .unwrap();
        assert_eq!(
            bits(&out),
            bits(&expected),
            "a refused block changed the canceller"
        );
    }
}
