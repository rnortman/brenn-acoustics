#!/usr/bin/env bash
# Copies Cargo.toml and the crate's sources into the test's temporary
# directory, where cargo may write target/ and Cargo.lock, and runs the pinned
# cargo's `check --offline --all-targets` on them with the pinned rustc and
# the toolchain's own sysroot from runfiles.
set -euo pipefail

cargo="$PWD/$CARGO"
export RUSTC="$PWD/$RUSTC"

# The sysroot is the directory four above any std rlib:
# <sysroot>/lib/rustlib/<triple>/lib/<name>.rlib.
std_rlib=""
for f in $RUST_STD; do
    case "$f" in *.rlib) std_rlib="$f"; break ;; esac
done
if [[ -z "$std_rlib" ]]; then
    echo "FAIL: no std rlib among the toolchain's stdlib files" >&2
    exit 1
fi
export RUSTFLAGS="--sysroot=$PWD/${std_rlib%/lib/rustlib/*}"

work="$TEST_TMPDIR/crate"
for f in $CARGO_SOURCES; do
    mkdir -p "$work/$(dirname "$f")"
    cp "$f" "$work/$f"
done

# Cargo reads config.toml from $CARGO_HOME and from .cargo/ in its working
# directory and every ancestor. Bazel's output base sits under the user's home
# directory, so an empty CARGO_HOME and a working directory of / keep a host
# cargo config (a rustc-wrapper, a target-dir, patches) out of the check.
export CARGO_HOME="$TEST_TMPDIR/cargo-home"
mkdir -p "$CARGO_HOME"
cd /

"$cargo" check --offline --all-targets --manifest-path "$work/Cargo.toml"
