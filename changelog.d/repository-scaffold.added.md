- First release of brenn-acoustics, acoustics algorithms for microphone
  arrays, free of I/O and of hardware dependencies, published under the
  Apache 2.0 licence.
- `delay::gcc_phat`: the delay between two channels by generalised
  cross-correlation with the phase transform, searched within a given number
  of frames either side and refined to a fraction of a frame, with the
  weighted correlation at the peak. Invalid input is an error: channels of
  different lengths, a search range not shorter than the channels, a sample
  that is not finite, or no signal.
- Bazel is the build. The Rust toolchain and every tool the checks use are
  pinned and downloaded by checksum. On linux x86_64 a fresh clone needs only
  bazelisk, git and make. `make check` is the one gate. It runs the build,
  the tests, rustfmt, clippy, a check of the Cargo manifest by the pinned
  cargo, buildifier, a lint of changelog and TODO fragments, and a secret
  scan. The crate also builds with Cargo.
- Secret scanning (brenn-scrub) runs before every commit and push once
  `make setup` has been run, and on every file Claude Code writes. GitHub
  Actions runs `make check` on every push to `main` and on every pull
  request.
- Changelog entries and TODO entries are kept as small fragment files under
  `changelog.d/` and `todo.d/`. A release gathers them into `CHANGELOG.md`
  and `TODO.md`.
