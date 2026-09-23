"""Correctness gate for the HipKittens attention forward.

Elementwise against the chunked fp32 reference in ../common.py, on the H3 shapes
and on sequence lengths that divide neither the KV block nor the Q tile.  Speed
is not gated here -- see sweep.sh -- but the aotriton number is printed in the
same process so the two are comparable at a glance.

    python3 test.py [--perf] [--heads H]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import D_HEAD, bench, check, flops, make_qkv, reference  # noqa: E402

import tk_kernel  # noqa: E402


def hk_attention(q, k, v, scale=0.0, causal=False):
    o = torch.empty_like(q)
    tk_kernel.dispatch_micro(q, k, v, o, float(scale), bool(causal))
    return o


# (label, b, h, n, kwargs).  h is small so the fp32 reference stays cheap; the
# kernel's head dimension is what the tiling depends on, and the head *count* is
# only a grid dimension.  kwargs go to both make_qkv and reference: `d` picks the
# head dimension, `h_kv` makes it GQA, `causal` masks above the diagonal.
SHAPES = [
    ("aligned  N=4096",  1, 4, 4096,  {}),
    ("aligned  N=16384", 1, 4, 16384, {}),
    ("h3 480p  N=49920", 1, 2, 49920, {}),
    ("batch    N=8192",  2, 4, 8192,  {}),
    # Exactly one Q tile at the shipped NUM_WARPS=12, and at NUM_WARPS=8 too.
    ("one q tile",       1, 4, 192,   {}),
    ("two q tiles",      1, 4, 384,   {}),
    # Neither the Q tile (192) nor the KV block (32) divides these.
    ("ragged   N=4097",  1, 4, 4097,  {}),
    ("ragged   N=5000",  1, 4, 5000,  {}),
    ("ragged   N=12345", 1, 4, 12345, {}),
    ("ragged   N=193",   1, 4, 193,   {}),

    # --- causal.  The interesting lengths are the ones where the diagonal does
    # not land on a block boundary, and the first Q tile, where most of the
    # workgroup's waves skip every block but the first.
    ("causal   N=4096",  1, 4, 4096,  dict(causal=True)),
    ("causal   N=4097",  1, 4, 4097,  dict(causal=True)),
    ("causal   N=5000",  1, 4, 5000,  dict(causal=True)),
    ("causal   one tile", 1, 4, 192,  dict(causal=True)),
    ("causal   N=193",   1, 4, 193,   dict(causal=True)),
    ("causal   batch",   2, 4, 2048,  dict(causal=True)),

    # --- GQA.  Ratios 4 and 8, plus ratio 1 written the GQA way, which is the
    # case an off-by-one in the head divide would still pass.
    ("gqa 4:1  N=4096",  1, 8, 4096,  dict(h_kv=2)),
    ("gqa 8:1  N=2048",  1, 8, 2048,  dict(h_kv=1)),
    ("gqa 1:1  N=1024",  1, 4, 1024,  dict(h_kv=4)),
    ("gqa+causal",       1, 8, 2048,  dict(h_kv=2, causal=True)),

    # --- head_dim 64.  Half the registers for q and the accumulator.
    ("d=64     N=4096",  1, 4, 4096,  dict(d=64)),
    ("d=64     N=4097",  1, 4, 4097,  dict(d=64)),
    ("d=64     causal",  1, 4, 2048,  dict(d=64, causal=True)),
    ("d=64     gqa",     1, 8, 2048,  dict(d=64, h_kv=2)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", action="store_true",
                    help="also time against SDPA on the H3 head count")
    args = ap.parse_args()

    p = torch.cuda.get_device_properties(0)
    print(f"gpu {p.gcnArchName}, {p.multi_processor_count} WGPs\n")

    fails = 0
    for label, b, h, n, kw in SHAPES:
        causal = kw.get("causal", False)
        q, k, v = make_qkv(b, h, n, d=kw.get("d", D_HEAD), h_kv=kw.get("h_kv"))
        try:
            got = hk_attention(q, k, v, causal=causal)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"{label:<20} b={b} h={h} n={n:>6}  LAUNCH {type(e).__name__}: {e}")
            fails += 1
            del q, k, v
            torch.cuda.empty_cache()
            continue
        ref = reference(q, k, v, causal=causal)
        ok, msg = check(f"{label:<20} b={b} h={h} n={n:>6}", got, ref)
        print(msg)
        fails += not ok
        del q, k, v, got, ref
        torch.cuda.empty_cache()

    if args.perf:
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel
        print()
        print(f"{'N':>7} {'hk ms':>10} {'hk TF':>8} {'sdpa ms':>10} {'sdpa TF':>8}")
        for n in (4096, 8192, 16384, 32768):
            q, k, v = make_qkv(1, 56, n)
            fl = flops(1, 56, n)
            hk = bench(lambda: hk_attention(q, k, v))
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                sd = bench(lambda: F.scaled_dot_product_attention(q, k, v))
            print(f"{n:>7} {hk:>10.3f} {fl/hk*1e-9:>8.1f} {sd:>10.3f} {fl/sd*1e-9:>8.1f}")
            del q, k, v
            torch.cuda.empty_cache()

    print(f"\n{'all shapes pass' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
