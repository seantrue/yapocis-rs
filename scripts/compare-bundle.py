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

    compare-bundle.py FRESH_DIR VENDORED_DIR SUBPATH...
"""

import difflib
import re
import sys
from pathlib import Path

SIG_DIGITS = 12
# A float literal not inside an identifier: digits with a decimal point and/or
# an exponent. Integers (indices, sizes) are left alone and compared exactly.
FLOAT = re.compile(r"(?<![\w.])(?:\d+\.\d*|\.\d+|\d+(?=[eE]))(?:[eE][-+]?\d+)?")


def normalise(text: str) -> str:
    return FLOAT.sub(lambda m: format(float(m.group(0)), f".{SIG_DIGITS}g"), text)


def files(root: Path, sub: str) -> dict[str, Path]:
    base = root / sub
    if base.is_file():
        return {sub: base}
    return {str(p.relative_to(root)): p for p in sorted(base.rglob("*")) if p.is_file()}


def main() -> int:
    fresh, vendored, subs = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    status = 0
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
