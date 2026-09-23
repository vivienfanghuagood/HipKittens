"""What a Radeon actually runs today for attention, how fast, and how we compare.

Three backends, all timed in one process:

  * the HipKittens kernel in ../fwd, through its torch op.
  * `torch.nn.functional.scaled_dot_product_attention`.  On gfx1100 both the
    FLASH and the EFFICIENT backend dispatch to the same aotriton kernel,
    `attn_fwd` -- confirmed with the profiler, not inferred.  This is the path
    diffusers and comfyUI take on Radeon, because both of them just call SDPA.
  * the Triton FA-2 in `triton_fa.py`, standing in for the Triton attention
    vLLM and SGLang ship for ROCm.

They are timed in this process, in this run, interleaved, because clock and
power state drift between runs and a number from a different run is not
comparable to one from this one.

    python3 bench_baselines.py [--check] [--big] [--causal]
"""

import argparse
import os
import sys
import warnings

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (BIG_SHAPES, SHAPES, bench, check, clocks, flops, make_qkv,
                    reference)

warnings.simplefilter("ignore")


def sdpa_aotriton(q, k, v, causal=False):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def hk(q, k, v, causal=False):
    """The HipKittens kernel, through the same torch op a framework would call.

    `attention`, not `scaled_dot_product_attention`: the drop-in falls back to
    torch for anything it cannot take, and a fallback timed here would be
    aotriton's number printed in our column.
    """
    import hk_attn
    return hk_attn.attention(q, k, v, is_causal=causal)


def backends():
    out = []
    try:
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "torch_ext"))
        import hk_attn
        if not hk_attn.loaded():
            raise RuntimeError("hk_attn_ext.so not built; run make in ../torch_ext")
        out.append(("hk", hk))
    except Exception as e:
        print(f"hk unavailable: {type(e).__name__}: {e}")
    out.append(("aotriton", sdpa_aotriton))
    try:
        from triton_fa import triton_attention
        out.append(("triton-fa", lambda q, k, v, causal=False:
                    triton_attention(q, k, v, causal=causal)))
    except Exception as e:
        print(f"triton-fa unavailable: {type(e).__name__}: {e}")
    return out


def which_kernel(fn, q, k, v):
    """Name the kernel that actually ran, so the table is not taking the
    backend enum's word for it."""
    from torch.profiler import ProfilerActivity, profile
    fn(q, k, v)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        fn(q, k, v)
        torch.cuda.synchronize()
    ks = [e.key for e in pr.key_averages() if e.device_time > 0]
    ks.sort(key=lambda kk: -max(e.device_time for e in pr.key_averages()
                                if e.key == kk))
    return ks[0][:40] if ks else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="also verify against the chunked fp32 reference")
    ap.add_argument("--big", action="store_true", help="include 720p/5s")
    ap.add_argument("--causal", action="store_true")
    args = ap.parse_args()

    bes = backends()
    print(f"gpu     {torch.cuda.get_device_properties(0).gcnArchName}, "
          f"{torch.cuda.get_device_properties(0).multi_processor_count} WGPs")
    print(f"torch   {torch.__version__}")
    print(f"clocks  {clocks()}")
    print(f"causal  {args.causal}\n")

    q, k, v = make_qkv(1, 8, 4096)
    for name, fn in bes:
        print(f"{name:10s} -> {which_kernel(fn, q, k, v)}")
    del q, k, v
    torch.cuda.empty_cache()
    print()

    hdr = f"{'shape':<14} {'B':>2} {'H':>3} {'N':>7}"
    for name, _ in bes:
        hdr += f" {name + ' ms':>13} {name + ' TF':>13}"
    print(hdr)

    shapes = SHAPES + (BIG_SHAPES if args.big else [])
    for label, b, h, n in shapes:
        try:
            q, k, v = make_qkv(b, h, n)
        except torch.OutOfMemoryError:
            print(f"{label:<14} {b:>2} {h:>3} {n:>7}  OOM allocating q/k/v")
            torch.cuda.empty_cache()
            continue
        row = f"{label:<14} {b:>2} {h:>3} {n:>7}"
        fl = flops(b, h, n, args.causal)
        outs = {}
        for name, fn in bes:
            try:
                iters = 3 if n > 40000 else 10
                ms = bench(lambda: fn(q, k, v, causal=args.causal), iters=iters)
                row += f" {ms:>13.3f} {fl / ms * 1e-9:>13.1f}"
                if args.check:
                    outs[name] = fn(q, k, v, causal=args.causal)
            except Exception as e:
                row += f" {type(e).__name__:>13} {'-':>13}"
        print(row)

        if args.check and outs:
            heads = list(range(min(h, 4)))
            ref = reference(q, k, v, causal=args.causal, heads=heads)
            for name, o in outs.items():
                print("    " + check(f"{label} {name}", o[:, heads], ref)[1])
            del ref
        del q, k, v, outs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
