# One correctness check plus two timings, small enough to sit inside a tiling
# sweep. test.py is the correctness harness and ../baselines/bench_baselines.py
# is the one that prints a comparable table; this only has to rank configs.
#
# The check comes first and is not optional: a tiling that spills has waits that
# no longer mean what they were written to mean, and the result is wrong rather
# than slow. A config that prints a good number and a wrong answer is the exact
# failure this file exists to catch.
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import D_HEAD, bench, check, flops, make_qkv, reference  # noqa: E402

import tk_kernel  # noqa: E402

# Small enough that the fp32 reference is cheap, ragged enough to exercise both
# tail paths (neither the Q tile nor the KV block divides 4097).
CHECK = (1, 4, 4097)
# H3's head count at two lengths. Override with
# QB_SHAPES="1x56x8192,1x56x32768" to point a sweep at a different regime.
SHAPES = [
    tuple(int(v) for v in s.split("x"))
    for s in os.environ.get("QB_SHAPES", "1x56x4096,1x56x16384").split(",")
]


def hk(q, k, v):
    o = torch.empty_like(q)
    tk_kernel.dispatch_micro(q, k, v, o, 0.0, False)
    return o


def timed(b, h, n):
    q, k, v = make_qkv(b, h, n)
    ms = bench(lambda: hk(q, k, v))
    del q, k, v
    torch.cuda.empty_cache()
    return flops(b, h, n) / ms * 1e-9


if __name__ == "__main__":
    if "--no-check" not in sys.argv:
        q, k, v = make_qkv(*CHECK)
        ok, msg = check("quickbench", hk(q, k, v), reference(q, k, v))
        del q, k, v
        torch.cuda.empty_cache()
        if not ok:
            print(f"WRONG ({msg})")
            sys.exit(1)
    print(" ".join(f"{timed(*s):.1f}" for s in SHAPES))
