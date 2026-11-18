- `array::LineArray`: a line of evenly spaced microphones, its spacing given
  as the time sound takes from one microphone to the next. Its module states
  the conventions every multi-microphone algorithm in the crate follows: one
  slice per microphone in their order along the line, and a direction as the
  angle from the line's axis, from 0 to pi radians.
- `beam::FixedBeams`: delay-and-sum beams steered to fixed angles, fed in
  blocks, with the same output however the signal is cut. A source on a
  beam's direction passes at gain 1, every beam lagging the microphones by
  the same number of frames, which `output_lag` reports. Its documentation
  states each beam's response and how little a short array separates beams
  at low frequencies. Bad angles and bad blocks are refused, and a refused
  block leaves the beams as they were.
- `doa::Direction`: the direction of the dominant source on a line of
  microphones, by steered response power with the phase transform, read
  every `hop` frames from the last `window`, fed in blocks with the same
  readings however the signal is cut. A reading carries no angle for
  silence or for sound that is not coherent across the microphones, and
  reports its peak either way. Bad configurations and bad blocks are
  refused, and a refused block leaves the estimate as it was.
- `beam::SelectedBeam`: one channel taken from fixed beams, moved to the beam
  nearest the direction estimate after `hold` readings in a row ask for it,
  with a linear crossfade of `fade` frames, fed in blocks with the same
  output however the signal is cut. It starts on the beam nearest broadside,
  and a source midway between two beams, silence and incoherent noise leave
  it where it is. Bad configurations and bad blocks are refused, and a
  refused block leaves the selection as it was.
