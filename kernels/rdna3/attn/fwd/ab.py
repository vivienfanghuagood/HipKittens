# In-process A/B of two builds of this kernel.
#
# The reason this file exists: bench() in ../common.py says that clock and power
# state drift between runs on this node, so two numbers from two processes are
# not comparable.  Building variant A, timing it, rebuilding as variant B and
# timing that -- which is what driving quickbench.py from a shell loop does --
# violates exactly that, and on this part it manufactured a 13-16% "gain" for
# PV_TILES=2 that the single-process baseline harness could not see at all.
#
# Usage:  python ab.py modA modB [modC ...]
# Each argument is an already-built extension module name.  Variants are
# interleaved A,B,...,A,B,... over repeated rounds so that any drift that
# remains is charged to all of them equally.
import importlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import bench, check, flops, make_qkv, reference  # noqa: E402

CAUSAL = os.environ.get("AB_CAUSAL", "0") == "1"
ROUNDS = int(os.environ.get("AB_ROUNDS", "5"))
# AB_CHECK=0 is for the ABLATE_* builds and nothing else.  Those delete a stage
# and are wrong on purpose, so the gate below would reject all of them -- but
# the stage attribution in README section 6 is a set of ratios between variants,
# and ratios measured in separate processes are the exact mistake this file
# exists to prevent.  For a real tiling change, leave the gate on.
CHECK = os.environ.get("AB_CHECK", "1") == "1"
SHAPES = [tuple(int(v) for v in s.split("x"))
          for s in os.environ.get("AB_SHAPES", "1x56x4096,1x56x16384").split(",")]


def main():
    names = sys.argv[1:]
    mods = [(n, importlib.import_module(n)) for n in names]

    # Correctness first, for every variant: a tiling that spills is wrong, not
    # slow, and a wrong variant that happens to be fast is the trap here.
    if CHECK:
        q, k, v = make_qkv(1, 4, 4097)
        ref = reference(q, k, v, causal=CAUSAL)
        for n, m in mods:
            o = torch.empty_like(q)
            m.dispatch_micro(q, k, v, o, 0.0, CAUSAL)
            ok, msg = check(n, o, ref)
            if not ok:
                print(f"{n}: WRONG ({msg})")
                return 1
        del q, k, v, ref
        torch.cuda.empty_cache()
    else:
        print("AB_CHECK=0: correctness gate skipped (ABLATE_* builds only)")

    print(f"causal={CAUSAL} rounds={ROUNDS}  (min over rounds, TFLOPs)")
    for b, h, n in SHAPES:
        q, k, v = make_qkv(b, h, n)

        def timed(m):
            def once():
                o = torch.empty_like(q)
                m.dispatch_micro(q, k, v, o, 0.0, CAUSAL)
            return bench(once)

        best = {nm: float("inf") for nm, _ in mods}
        for _ in range(ROUNDS):
            for nm, m in mods:
                best[nm] = min(best[nm], timed(m))
        f = flops(b, h, n) / (2 if CAUSAL else 1)
        cells = "  ".join(f"{nm}={f / best[nm] * 1e-9:6.2f}" for nm, _ in mods)
        base = f / best[names[0]] * 1e-9
        rel = "  ".join(f"{(f / best[nm] * 1e-9) / base:5.3f}x" for nm, _ in mods)
        print(f"  b={b} h={h} n={n:6d}  {cells}   |  {rel}")
        del q, k, v
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
