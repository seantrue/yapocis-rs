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

# Float literals are compared to 12 significant digits: numpy can round the
# rendered Gaussian weights differently in the last bit on different CPUs.
status=0
python3 "$here/scripts/compare-bundle.py" "$tmp" "$here/crates/yapocis/bundle" \
    manifest.json ocl metal cuda fixtures || status=1
if [ "$status" -eq 0 ]; then
    echo "bundle is up to date"
fi
exit "$status"
