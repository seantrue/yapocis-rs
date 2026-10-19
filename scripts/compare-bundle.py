#!/usr/bin/env python3
"""Compare a freshly exported bundle with the vendored one, tolerating float noise.

The Gaussian weights baked into the kernel sources (and recorded in the
manifest and fixtures) are computed by numpy when the bundle is rendered, and
numpy's SIMD `exp`/`sum` can round the last bit differently on different CPUs,
so the same Yapocis3 commit renders `0.0028602900165330338` on one hosted
runner and `0.0028602900165330333` on another. Every float literal (a number
with a decimal point or an exponent) is therefore rounded to 12 significant
digits before comparing; the kernels compute in float32, so nothing that
reaches a device is lost. Any other difference, and any missing or extra file,
is drift.

The manifest records a SHA-256 per rendered source, which that noise also
changes. Those hashes are left out of the comparison (the sources they cover
are compared directly, with the tolerance); instead every vendored hash must
match the vendored file it names.

    compare-bundle.py FRESH_DIR VENDORED_DIR SUBPATH...
"""

import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

SIG_DIGITS = 12
# A float literal not inside an identifier: digits with a decimal point and/or
# an exponent. Integers (indices, sizes) are left alone and compared exactly.
FLOAT = re.compile(r"(?<![\w.])(?:\d+\.\d*|\.\d+|\d+(?=[eE]))(?:[eE][-+]?\d+)?")
SHA256 = re.compile(r'("sha256": )"[0-9a-f]{64}"')


def normalise(text: str) -> str:
    text = SHA256.sub(r'\1"<sha256>"', text)
    return FLOAT.sub(lambda m: format(float(m.group(0)), f".{SIG_DIGITS}g"), text)


def source_hashes(node):
    """Every (path, sha256) pair recorded anywhere in the manifest."""
    if isinstance(node, dict):
        if isinstance(node.get("path"), str) and isinstance(node.get("sha256"), str):
            yield node["path"], node["sha256"]
        for v in node.values():
            yield from source_hashes(v)
    elif isinstance(node, list):
        for v in node:
            yield from source_hashes(v)


def check_vendored_hashes(vendored: Path) -> int:
    status = 0
    pairs = list(source_hashes(json.loads((vendored / "manifest.json").read_text())))
    for path, want in pairs:
        f = vendored / path
        got = hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else "missing"
        if got != want:
            print(f"bundle drift: manifest sha256 for {path} does not match the vendored file ({got})")
            status = 1
    if not pairs:
        print("bundle drift: no source hashes found in the vendored manifest")
        status = 1
    return status


def files(root: Path, sub: str) -> dict[str, Path]:
    base = root / sub
    if base.is_file():
        return {sub: base}
    return {str(p.relative_to(root)): p for p in sorted(base.rglob("*")) if p.is_file()}


def main() -> int:
    fresh, vendored, subs = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    status = check_vendored_hashes(vendored) if "manifest.json" in subs else 0
    for sub in subs:
        a, b = files(fresh, sub), files(vendored, sub)
        for name in sorted(a.keys() - b.keys()):
            print(f"bundle drift: {name} is new (not vendored)")
            status = 1
        for name in sorted(b.keys() - a.keys()):
            print(f"bundle drift: {name} is vendored but no longer exported")
            status = 1
        for name in sorted(a.keys() & b.keys()):
            x, y = a[name].read_text(), b[name].read_text()
            if x == y:
                continue
            nx, ny = normalise(x), normalise(y)
            if nx == ny:
                continue
            print(f"bundle drift in {name}:")
            diff = difflib.unified_diff(
                ny.splitlines(), nx.splitlines(), "vendored", "exported", lineterm="", n=1
            )
            for i, line in enumerate(diff):
                if i == 40:
                    print("  ...")
                    break
                print(f"  {line}")
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
