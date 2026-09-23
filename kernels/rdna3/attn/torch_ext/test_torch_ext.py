"""Correctness and speed for the torch operator and the SDPA drop-in.

Two different things are being checked.  The op has to agree with the fp32
reference elementwise on the shapes it claims.  The wrapper has to route every
*other* call to torch instead of guessing -- and a fallback that silently
returned a wrong answer and a fallback that never happened would both look like
"fast and correct" in a naive test, so each fallback branch is asserted twice:
`supported()` must say no, and the result must still match the reference.

    python3 test_torch_ext.py [--perf]
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import D_HEAD, bench, check, flops, make_qkv, reference  # noqa: E402

import hk_attn  # noqa: E402

# (label, b, h, n, kwargs).  Same set as fwd/test.py: the H3 shapes, sequence
# lengths that divide neither the Q tile nor the KV block, and the coverage the
# kernel grew in phase E.  h is small so the fp32 reference stays cheap -- head
# count is only a grid dimension.  The kwargs are shared between make_qkv, the
# reference and the drop-in, so a case that is routed wrong is also checked
# wrong and cannot pass by accident.
SHAPES = [
    ("aligned  N=4096",  1, 4, 4096,  {}),
    ("aligned  N=16384", 1, 4, 16384, {}),
    ("h3 480p  N=49920", 1, 2, 49920, {}),
    ("batch    N=8192",  2, 4, 8192,  {}),
    # Exactly one Q tile at the shipped NUM_WARPS=12, and at NUM_WARPS=8 too.
    ("one q tile",       1, 4, 192,   {}),
    ("two q tiles",      1, 4, 384,   {}),
    ("ragged   N=4097",  1, 4, 4097,  {}),
    ("ragged   N=5000",  1, 4, 5000,  {}),
    ("ragged   N=12345", 1, 4, 12345, {}),
    ("ragged   N=193",   1, 4, 193,   {}),
    ("causal   N=4096",  1, 4, 4096,  dict(causal=True)),
    ("causal   N=5000",  1, 4, 5000,  dict(causal=True)),
    ("causal   one tile", 1, 4, 192,  dict(causal=True)),
    ("gqa 4:1  N=4096",  1, 8, 4096,  dict(h_kv=2)),
    ("gqa 8:1  N=2048",  1, 8, 2048,  dict(h_kv=1)),
    ("gqa+causal",       1, 8, 2048,  dict(h_kv=2, causal=True)),
    ("d=64     N=4096",  1, 4, 4096,  dict(d=64)),
    ("d=64     causal",  1, 4, 2048,  dict(d=64, causal=True)),
    ("d=64     gqa",     1, 8, 2048,  dict(d=64, h_kv=2)),
]


def torch_ref(q, k, v, **kw):
    """torch's own answer in fp32, for the fallback cases.

    Materializes the score matrix, so it is only used at n=512 where that is
    150 KB per head.  The big shapes go through common.reference, which chunks.
    """
    kw.pop("dropout_p", None)
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q.detach().float(), k.detach().float(),
                                              v.detach().float(), **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", action="store_true")
    args = ap.parse_args()

    p = torch.cuda.get_device_properties(0)
    print(f"gpu {p.gcnArchName}, {p.multi_processor_count} WGPs")
    print(f"extension loaded: {hk_attn.loaded()}\n")
    if not hk_attn.loaded():
        return 1

    fails = 0

    # --- the supported shapes, through the drop-in -----------------------
    for label, b, h, n, kw in SHAPES:
        causal = kw.get("causal", False)
        gqa = kw.get("h_kv") is not None
        q, k, v = make_qkv(b, h, n, d=kw.get("d", D_HEAD), h_kv=kw.get("h_kv"))
        call = dict(is_causal=causal, enable_gqa=gqa)
        if not hk_attn.supported(q, k, v, **call):
            print(f"{label:<20} b={b} h={h} n={n:>6}  FAIL should be supported: "
                  f"{hk_attn._why(q, k, v, None, 0.0, causal, gqa)}")
            fails += 1
            del q, k, v
            torch.cuda.empty_cache()
            continue
        got = hk_attn.scaled_dot_product_attention(q, k, v, **call)
        ok, msg = check(f"{label:<20} b={b} h={h} n={n:>6}", got,
                        reference(q, k, v, causal=causal))
        print(msg)
        fails += not ok
        del q, k, v, got
        torch.cuda.empty_cache()

    # --- a non-contiguous input, which is the layout frameworks produce --
    # Attention modules project to (B, N, H, D) and transpose; the result is
    # never contiguous.  The wrapper must copy rather than read it wrong.
    b, h, n = 1, 4, 4096
    qkv = [torch.randn(b, n, h, D_HEAD, device="cuda", dtype=torch.bfloat16)
           .transpose(1, 2) for _ in range(3)]
    assert not qkv[0].is_contiguous()
    assert hk_attn.supported(*qkv), "a non-contiguous input should still be supported"
    got = hk_attn.scaled_dot_product_attention(*qkv)
    ok, msg = check(f"{'non-contiguous':<20} b={b} h={h} n={n:>6}", got, reference(*qkv))
    print(msg)
    fails += not ok
    # ... and the raw op must refuse it rather than produce that same answer.
    try:
        torch.ops.hk_attn.fwd(qkv[0], qkv[1], qkv[2], 0.0, False)
        print(f"{'raw op took non-contiguous':<34} FAIL it must refuse")
        fails += 1
    except Exception:
        print(f"{'raw op rejects non-contiguous':<34} raises, as it should")
    del qkv, got
    torch.cuda.empty_cache()

    # --- a non-default scale --------------------------------------------
    q, k, v = make_qkv(1, 4, 1024)
    got = hk_attn.scaled_dot_product_attention(q, k, v, scale=0.05)
    ok, msg = check(f"{'scale=0.05':<34}", got, reference(q, k, v, scale=0.05))
    print(msg)
    fails += not ok
    del q, k, v, got

    # --- every fallback branch: not supported, and still right -----------
    print()
    b, h, n = 1, 4, 512
    q, k, v = make_qkv(b, h, n)
    q32 = torch.randn(b, h, n, 32, device="cuda", dtype=torch.bfloat16)
    kv2 = torch.randn(b, 2, n, D_HEAD, device="cuda", dtype=torch.bfloat16)
    kv_short = torch.randn(b, h, 256, D_HEAD, device="cuda", dtype=torch.bfloat16)
    cases = [
        # A bool mask, not an additive one: dtype-agnostic, so the reference can
        # upcast q/k/v to fp32 and reuse it. tril keeps every row non-empty.
        ("attn_mask",     (q, k, v),                     dict(attn_mask=torch.ones(
                                                             n, n, device="cuda",
                                                             dtype=torch.bool).tril())),
        ("fp16",          tuple(t.half() for t in (q, k, v)), {}),
        ("D=32",          (q32, q32.clone(), q32.clone()), {}),
        # n_kv != n_q. The kv loop is bounded by q's length, so this is the one
        # shape the kernel would read wrong rather than fail on.
        ("cross attention", (q, kv_short, kv_short.clone()), {}),
        ("requires_grad", (q.clone().requires_grad_(True), k, v), {}),
        ("N < Q tile",    tuple(t[:, :, :64] for t in (q, k, v)), {}),
    ]
    for name, t, kw in cases:
        if hk_attn.supported(*t, **kw):
            print(f"{'fallback ' + name:<34} FAIL reported as supported")
            fails += 1
            continue
        got = hk_attn.scaled_dot_product_attention(*t, **kw)
        ok, msg = check(f"{'fallback ' + name:<34}", got.detach(), torch_ref(*t, **kw))
        print(msg)
        fails += not ok

    # Mismatched head counts without enable_gqa is torch's own error, and the
    # point of falling back is that the caller gets torch's message rather than
    # ours -- so the check is that it raises, not that it computes.
    if hk_attn.supported(q, kv2, kv2):
        print(f"{'fallback GQA not enabled':<34} FAIL reported as supported")
        fails += 1
    else:
        try:
            hk_attn.scaled_dot_product_attention(q, kv2, kv2)
            print(f"{'fallback GQA not enabled':<34} FAIL torch should have raised")
            fails += 1
        except Exception:
            print(f"{'fallback GQA not enabled':<34} torch raises, as it should")

    # dropout is stochastic, so only the routing is checkable.
    assert not hk_attn.supported(q, k, v, dropout_p=0.1)
    hk_attn.scaled_dot_product_attention(q, k, v, dropout_p=0.1)
    print(f"{'fallback dropout':<34} routed to torch")

    # --- the loud entry point must stay loud -----------------------------
    # attention() is what benchmarks call: silently running torch's kernel
    # there would be measured as a win for ours.
    try:
        hk_attn.attention(q32, q32, q32)
        print(f"{'attention() accepted D=32':<34} FAIL it must raise")
        fails += 1
    except Exception:
        print(f"{'attention() rejects D=32':<34} raises, as it should")
    del q, k, v, q32, kv2, kv_short
    torch.cuda.empty_cache()

    # --- speed: the drop-in and aotriton, one process, one run -----------
    if args.perf:
        print()
        print(f"{'N':>7} {'hk ms':>10} {'hk TF':>8} {'sdpa ms':>10} {'sdpa TF':>8} {'x':>6}")
        for n in (4096, 8192, 16384, 32768):
            q, k, v = make_qkv(1, 56, n)
            fl = flops(1, 56, n)
            hk = bench(lambda: hk_attn.scaled_dot_product_attention(q, k, v))
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                sd = bench(lambda: F.scaled_dot_product_attention(q, k, v))
            print(f"{n:>7} {hk:>10.3f} {fl/hk*1e-9:>8.1f} {sd:>10.3f} "
                  f"{fl/sd*1e-9:>8.1f} {sd/hk:>6.2f}")
            del q, k, v
            torch.cuda.empty_cache()

    print(f"\n{'all pass' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
