- A first echo canceller, `echo::Canceller`: given what was sent to the
  loudspeaker (the reference) and what one microphone heard, it returns the
  microphone's signal with the loudspeaker's echo removed. It adapts as it
  runs (normalised least mean squares), keeps adapting when someone talks
  over the playback, though more slowly, and recovers when the echo path
  changes. It is fed in blocks of any size, with the same output however the
  signal is cut. Its output lags the microphone by a configured number of
  frames, which `output_lag` reports. Bad settings and bad blocks (mismatched
  lengths, a sample that is not finite) are refused, and a refused block
  leaves the canceller as it was.
- `delay::reference_delay`: estimates how far the echo of the reference sits
  from it in a microphone signal, in whole frames, ahead or behind and of
  either polarity, with a measure of how clearly it stands out. It refuses
  signals that cannot support an estimate rather than guessing.
- `filter::HighPass`: a streaming high-pass filter (fourth-order
  Butterworth), for removing hum and rumble before delay estimation and echo
  cancellation.
