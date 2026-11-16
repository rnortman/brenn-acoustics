# brenn-acoustics

Acoustics algorithms for brenn's microphone arrays: echo cancellation,
beamforming and direction of arrival, built from small primitives that can be
tested on signals generated in code.

## Charter

- **Free of I/O and of hardware dependencies.** Every function takes samples
  and returns results. Nothing in this repository reads or writes a file, a
  device or the network, binds to another language or parses a command line.
  Tuning for a CPU or accelerator architecture is allowed; firmware for a
  particular board is not.
- **A leaf.** It depends on no other brenn repository. brenn's device firmware
  and host code are expected to depend on it, and the experiments that judge
  the algorithms on recorded captures call it from their own repository.
- **Plain scalar `f32`** for now. Speed comes later, once there is something
  worth speeding up.

## Status

One primitive: `brenn_acoustics::delay::gcc_phat`, the delay between two
channels by generalised cross-correlation with the phase transform, to a
fraction of a frame.

## Building

Bazel is the build. On linux x86_64 a fresh clone needs bazelisk (as
`bazel`), git and make:

```sh
make setup   # once per clone: wires the git hooks
make check   # the gate
```

`make help` lists every target. The crate also has a `Cargo.toml`, for Cargo
consumers; a test in the default check holds it to the Bazel build.

## Licence

Apache-2.0; see `LICENSE` and `NOTICE`.
