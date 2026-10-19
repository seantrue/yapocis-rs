#!/usr/bin/env bash
# Regenerate the kernel bundle with the Python package and diff it against the
# vendored copy. Fails when interfaces or templates changed without a regen.
#
# Usage: scripts/check-bundle.sh [path/to/yapocis-tools]
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
tools="${1:-yapocis-tools}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

"$tools" bundle --out "$tmp" --fixtures >/dev/null

status=0
for sub in manifest.json ocl metal cuda fixtures; do
    if ! diff -r "$tmp/$sub" "$here/crates/yapocis/bundle/$sub"; then
        echo "bundle drift in $sub" >&2
        status=1
    fi
done
if [ "$status" -eq 0 ]; then
    echo "bundle is up to date"
fi
exit "$status"
