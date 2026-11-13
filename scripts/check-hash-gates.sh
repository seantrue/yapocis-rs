#!/bin/sh
# The hash gates (the commonkit hash facade, D10): every digest goes through commonkit::digest.
#  1. hash crates appear in the build only under commonkit;
#  2. bindings/python has exactly one commonkit, 0.2.x;
#  3. no source outside tests and hash-allowlist.txt names an algorithm or a hash library.
set -u
cd "$(dirname "$0")/.."
status=0
features=yapocis/ocl,yapocis/cuda,yapocis/rustcpu
for c in md-5 md5 sha1 sha2 blake3; do
  for m in Cargo.toml bindings/python/Cargo.toml; do
    f=""; [ "$m" = Cargo.toml ] && f="--workspace --features $features"
    # shellcheck disable=SC2086
    out=$(cargo tree --manifest-path "$m" --locked $f -e normal,build -i "$c" --depth 1 --prefix none 2>/dev/null) || continue
    bad=$(printf '%s\n' "$out" | sed 1d | grep -v '^commonkit v' | grep -v '^$')
    if [ -n "$bad" ]; then echo "hash gate: $c is used outside commonkit in $m: $bad"; status=1; fi
  done
done
n=$(cargo tree --manifest-path bindings/python/Cargo.toml --locked -i commonkit -e normal,build --depth 0 | grep -c '^commonkit v0\.2\.')
[ "$n" = 1 ] || { echo "hash gate: bindings/python has $n commonkit v0.2 lines, want 1"; status=1; }
pat='hashlib|import blake3|blake3::|sha2::|sha1::|md5::|Sha256::|Algo::Md5|Algo::B3|fixed::|build::both|"md5:|"b3:'
for f in $(git ls-files '*.rs' '*.py' | grep -v '/tests/' | grep -vxF -f hash-allowlist.txt); do
  # Rust: the in-file `#[cfg(test)]` module is a test.
  hits=$(awk '/^#\[cfg\(test\)\]/{exit} {print FILENAME":"FNR": "$0}' "$f" | grep -E "$pat")
  if [ -n "$hits" ]; then echo "$hits"; status=1; fi
done
[ $status = 0 ] && echo "hash gates: ok"
exit $status
