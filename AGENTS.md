# brenn-acoustics — agent instructions

## What this is

brenn-acoustics holds brenn's acoustics algorithms for microphone arrays: the
pieces of echo cancellation, beamforming and direction of arrival. They are
free of I/O and of hardware dependencies. Tuning for a CPU or accelerator
architecture is allowed; firmware for a particular board is not. Charter and
status: `README.md`. Read it first.

## Boundaries

- This repository depends on nothing in brenn-pod, brenn-lab or
  brenn-pod-experiments. brenn-pod is expected to depend on it, so the
  opposite would be a cycle.
- Nothing in it reads or writes a file, a device or the network, binds to
  Python or any other language, or parses a command line, tests included.
  Tests run on signals generated in code. A binding or a harness for this
  crate lives in the repository that needs it.
- Samples are plain scalar `f32` for now: no SIMD, no fixed point, no
  embedded target.
- Invalid input is an error, never coerced: a function states what it
  accepts and refuses the rest.

## Build: Bazel, and only Bazel

Bazel is the build. The Makefile is a frontend whose every target is a
spelled-out `bazel` invocation; there is no target that runs `cargo` on the
host, and the gate never will.

- Every toolchain and every tool the gate runs is pinned in `MODULE.bazel`
  and fetched by checksum. On linux x86_64 the only host prerequisites are
  bazelisk (as `bazel`), git and make, on a base system that has `bash`,
  `grep`, `sed` and the coreutils. On linux aarch64 the C toolchain has no
  pinned sysroot, so building the gate tools also needs the host's libc
  headers and libraries.
- Nothing in a build or test action may reach for a host-installed program.
  If a tool is genuinely impractical to bring in through Bazel, pick a
  different tool, and if that is not possible either, surface it as a
  decision rather than adding a host dependency.
- The crate is `//:brenn_acoustics`, in the root `BUILD.bazel`. Its sources
  are listed there by name: a new source file is added to `SRCS`. The root
  package loads only rules_rust, so a module depending on this one loads it
  without the gate's tools; the gate's other targets live under `tools/`.
- `MODULE.bazel` declares rules_rust and the Rust toolchain as ordinary
  dependencies and everything else as a dev dependency, so a dependant sees
  a Rust library and nothing more.
- The crate has no third-party dependency. It owns its small FFT, private to
  the crate, so replacing it changes no interface.
- `Cargo.toml` exists for Cargo consumers. `//tools/cargo_check:cargo_check_test`
  runs the pinned toolchain's `cargo check --offline --all-targets` over it
  and the sources the Bazel build lists, so the two descriptions stay equal.
  That test is the only place cargo runs.
- The repository's Rust is held to rustfmt by a test in the default check and
  to clippy, every warning an error, under `make check`'s lint config.
- The two gate tools, `//tools/scrub` (the secret scanner) and
  `//tools/allthechanges` (the changelog and TODO fragment tool), are built
  from their crates, each hub pinned by its own `Cargo.lock` and
  `cargo-bazel-lock.json`. `make repin` regenerates both; after a tool bump,
  run it and commit every lock file that changed.

## Commands

| Command | What it does |
|---|---|
| `make check` | The gate: scrub kit check, fragment lint, build, the unit tests, rustfmt, clippy, the Cargo manifest check, buildifier, tree scan |
| `make build` | Build every target |
| `make test` | Every test, without the lint aspect (the inner loop) |
| `make fix` | rustfmt and buildifier the tree in place |
| `make repin` | Regenerate the two gate tools' crate locks |
| `make todo-gather` | Fold `todo.d/` fragments into `TODO.md` |
| `make changes ARGS=...` | Run allthechanges: `ARGS="query"`, `ARGS="todo query"`, `ARGS="gather --version X.Y.Z --dry-run"` |
| `make setup` | Lay the scrub kit and wire git at `.githooks` (once per clone; `make check` fails until it has run) |
| `make scrub-tree` | Whole-tree secret sweep |
| `make clean` | Drop Bazel's outputs for this workspace |

## Gates

- A commit scans the staged change for secrets and then runs `make check`,
  which also lints every `changelog.d/` and `todo.d/` fragment.
- A push scans the range being pushed.
- CI runs `make check` on every push to `main` and every pull request, with
  the module lockfile in error mode, so a resolution nobody committed is a red
  build.
- The Claude Code write hook scrubs every write before it lands and blocks on
  any failure, including a scanner that fails to start.

Every scan runs `bazel run //tools/scrub:brenn-scrub`. Treat the local hooks
as the gate and CI as a backstop, not the other way round: a CI run says
nothing about the commit sitting in your working tree.

If a gate blocks a write or a commit, **surface it**. Never route around it.
The gate being wrong is a thing that happens and is worth reporting; a
bypassed gate is not recoverable once it has been pushed.

One case is not a route-around. The write hook builds the scanner from the
working tree, so an edit that breaks a file the scanner's build reads (for
example `MODULE.bazel`, `.bazelversion`, `.bazelrc`, or anything under
`tools/scrub/`) makes the hook fail on every later write, the repair included.
When the hook's output shows that the scanner could not be built, and not that
it ran and reported a finding, and the failure follows an edit you made,
read `git diff HEAD -- <path>`, restore the file you broke to its last
committed content with `git restore --source=HEAD -- <path>`, then make every
change that diff showed again with the editing tool, not only the one that
broke it. Restore nothing else, write nothing by any other route, and say
what happened in your report of the work, quoting the hook's output. If no edit
of yours is behind the failure, or the hook still fails after the restore,
stop and report. To stay out of the lockout, make every edit to such a file
leave the build loadable: add a `load` before the rule that uses it.

Text an agent writes by hand into a file in this repository goes through its
file-editing tools, so the write hook scans it. What the repository's own
commands generate or rewrite (lock files, the files `make setup` lays,
formatter and fixer output) is not written by hand.

## Principles

- Fail fast. Invalid input is rejected, not coerced. No fallbacks that hide a
  missing authority.
- No over-engineering, no premature abstraction, no hypothetical features.
  Scope discipline: do what was asked.
- Port only from permissively licensed references.

## Code standards

- Rust, edition 2024, the pinned toolchain.
- `make check` must pass: rustfmt, clippy with every warning an error,
  buildifier, every test.
- No unused code. No backwards-compatibility shims.
- Default branch: `main`.

## Changelog

Every user-visible change adds a fragment, `changelog.d/<slug>.<section>.md`,
where `<section>` is one of `added`, `changed`, `deprecated`, `removed`,
`fixed`, `security`. A fragment is a Markdown list: its first line starts with
`- `, continuation lines are indented two spaces. Never edit `CHANGELOG.md`;
`make changes ARGS="gather --version X.Y.Z"` gathers the fragments into it at a
release. `make changes ARGS="query"` lints and lists what is pending.

## TODO system

Two pieces that stay in sync:

- The master list: `TODO.md` at the repo root plus the fragments under
  `todo.d/` not yet gathered into it. Each entry has a slug, a description,
  and the deferral context.
- `TODO(slug)` comments in code, marking the spot where the work happens.

A new entry is a fragment, `todo.d/<slug>.md`: its first line is text, it has
no ATX heading outside a code fence, and every fence is closed. Never add an
entry by editing `TODO.md`. `make todo-gather` folds the fragments into
`TODO.md`; between gathers `make changes ARGS="todo query"` lists them.
Removing a done TODO deletes its fragment if it still is one, else its
`` ## `<slug>` `` entry in `TODO.md`, and its comment.

Slugs are the join key; adding a TODO requires both halves. Don't use TODOs for
vague aspirations. Every TODO describes a concrete thing, in a place where
"done" is obvious.

## Public tree

Everything tracked here is published. `TODO.md`, `CHANGELOG.md`, their
fragments under `todo.d/` and `changelog.d/`, commit messages and every code
comment are public writing: no internal host names, no absolute paths into
other trees, no references to documents a public reader cannot follow. Design
documents live outside this repository and are never cited from code.
