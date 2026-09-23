"""Verify and benchmark torch.ops.hk_dist against what a framework does today.

    HIP_VISIBLE_DEVICES=0,3 timeout 1200 torchrun --nproc_per_node=2 test_torch_ext.py

The reference is F.linear over this rank's K-slice followed by
dist.all_reduce -- literally the body of vLLM's RowParallelLinear.forward, on
RCCL. Every shape is checked before it is timed; a perf number for a kernel that
does not compute the right thing is worse than no number.

Two measurement rules this file exists to obey, both learned the hard way:

  * Weights are rotated at decode sizes. This card has 96 MB of Infinity Cache
    and the decode weights are 31-89 MB, so hammering one copy measures the
    cache and reports bandwidths above the HBM peak. --rotate picks how many
    copies to cycle through.

  * The peer bandwidth of this node is bimodal per process launch -- about
    15.5 GB/s or about 25 GB/s, nothing between -- so a fused time from one run
    and a collective time from another describe different hardware. Everything
    compared here is measured in one process, in one run, and the run prints
    which mode it landed in.
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hk_dist   # noqa: E402

# Qwen3-27B-class, TP-sharded. K is the un-sharded reduction dim; each rank
# owns K // world of it. N is the un-sharded output width.
#   attn_out  o_proj:    K = num_heads * head_dim = 6144, N = hidden = 5120
#   mlp_down  down_proj: K = intermediate = 17408,        N = hidden = 5120
OPS = [("attn_out", 6144, 5120), ("mlp_down", 17408, 5120)]
DECODE_M = [1, 2, 4, 8, 16, 32]
PREFILL_M = [2048, 4096, 8192]


def sync():
    torch.cuda.synchronize()
    dist.barrier()


def timeit(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    sync()
    return (t1 - t0) * 1e3 / iters


def rel_err(got, ref):
    d = (got.float() - ref.float()).abs().max().item()
    s = ref.float().abs().max().item()
    # A reference with no magnitude makes any ratio meaningless, and a
    # separable-generator bug once produced exactly that -- every output ~0 and
    # a perfect score. Fail loudly instead.
    if s < 1e-3:
        return float("inf")
    return d / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rotate", type=int, default=4,
                    help="weight copies to cycle at decode sizes, to defeat the "
                         "96 MB Infinity Cache")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--decode-iters", type=int, default=100)
    ap.add_argument("--tol", type=float, default=5e-2)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", local_rank)
    torch.manual_seed(1234 + rank)

    max_m = max(PREFILL_M)
    max_n = max(n for _, _, n in OPS)
    hk_dist.init(max_tokens=max_m, hidden_size=max_n)

    if rank == 0:
        name = torch.cuda.get_device_name(0)
        print(f"=== torch.ops.hk_dist, {world} ranks x {name}, "
              f"torch {torch.__version__} ===\n")
        print("ref  = F.linear(x, w) + dist.all_reduce  (what a framework runs "
              "today, on RCCL)")
        print("gemm = F.linear alone, no collective: hipBLASLt's time for the "
              "same matmul.")
        print("       f/g is the fused op against it, so it mixes two effects "
              "-- how much of")
        print("       the collective got hidden, and how our GEMM compares to "
              "hipBLASLt's.")
        print("       gemm_ar_mp.hip measures the first alone, against our own "
              "GEMM.\n")
        print(f"{'phase':<8}{'op':<10}{'M':>6}{'K_loc':>7}{'N':>6}"
              f"{'gemm_ms':>9}{'ref_ms':>9}{'fus_ms':>9}"
              f"{'speedup':>9}{'f/g':>7}{'rel':>10}  check")

    ok_all = True
    rows = []
    for phase, Ms, iters in (("decode", DECODE_M, args.decode_iters),
                             ("prefill", PREFILL_M, args.iters)):
        for op, K, N in OPS:
            K_local = K // world
            rot = args.rotate if phase == "decode" else 1
            ws = [torch.randn(N, K_local, device=dev, dtype=torch.bfloat16) / 32
                  for _ in range(rot)]
            for M in Ms:
                x = torch.randn(M, K_local, device=dev, dtype=torch.bfloat16) / 32
                w = ws[0]

                why = hk_dist.why_unsupported(M, N, K_local)
                if why:
                    if rank == 0:
                        print(f"{phase:<8}{op:<10}{M:>6}{K_local:>7}{N:>6}"
                              f"{'':>43}  -- {why}")
                    continue

                # Correctness first, on one copy, outside any timing.
                ref = F.linear(x, w)
                dist.all_reduce(ref)
                got = hk_dist.linear_allreduce(x, w)
                r = rel_err(got, ref)
                # bf16 rounds twice here (product and sum) for about 1.2e-2 over
                # a K this long; 5e-2 keeps 4x of headroom while staying 20x
                # below the O(1) a mis-routed tile produces.
                ok = r < args.tol
                ok_all &= ok

                i = [0]

                def next_w():
                    i[0] = (i[0] + 1) % rot
                    return ws[i[0]]

                def run_gemm():
                    F.linear(x, next_w())

                def run_ref():
                    y = F.linear(x, next_w())
                    dist.all_reduce(y)

                def run_fused():
                    hk_dist.linear_allreduce(x, next_w())

                t_gemm = timeit(run_gemm, iters)
                t_ref = timeit(run_ref, iters)
                t_fus = timeit(run_fused, iters)

                if rank == 0:
                    print(f"{phase:<8}{op:<10}{M:>6}{K_local:>7}{N:>6}"
                          f"{t_gemm:>9.3f}{t_ref:>9.3f}{t_fus:>9.3f}"
                          f"{t_ref / t_fus:>8.2f}x{t_fus / t_gemm:>7.2f}"
                          f"{r:>10.2e}  {'ok' if ok else 'MISMATCH'}")
                rows.append((phase, op, M, t_ref / t_fus, t_fus / t_gemm, ok))
            del ws

    # The reduce-scatter form, on the shapes it accepts. Its output is local, so
    # unlike the all-reduce form it does not pay a copy out of the symmetric
    # heap -- worth measuring separately rather than assuming it tracks.
    if rank == 0:
        print(f"\n{'--- reduce-scatter (sequence-parallel form) ---':<60}")
        print(f"{'phase':<8}{'op':<10}{'M':>6}{'K_loc':>7}{'N':>6}"
              f"{'ref_ms':>9}{'fus_ms':>9}{'speedup':>9}{'rel':>10}  check")
    for op, K, N in OPS:
        K_local = K // world
        w = torch.randn(N, K_local, device=dev, dtype=torch.bfloat16) / 32
        for M in PREFILL_M:
            why = hk_dist.why_unsupported(M, N, K_local, "reduce_scatter")
            if why:
                if rank == 0:
                    print(f"{'prefill':<8}{op:<10}{M:>6}{K_local:>7}{N:>6}"
                          f"{'':>37}  -- {why}")
                continue
            x = torch.randn(M, K_local, device=dev, dtype=torch.bfloat16) / 32
            shard = M // world

            def run_ref_rs():
                y = F.linear(x, w)
                out = torch.empty(shard, N, device=dev, dtype=torch.bfloat16)
                dist.reduce_scatter_tensor(out, y)
                return out

            ref = run_ref_rs()
            got = hk_dist.linear_reducescatter(x, w)
            r = rel_err(got, ref)
            ok = r < args.tol
            ok_all &= ok
            t_ref = timeit(run_ref_rs, args.iters)
            t_fus = timeit(lambda: hk_dist.linear_reducescatter(x, w), args.iters)
            if rank == 0:
                print(f"{'prefill':<8}{op:<10}{M:>6}{K_local:>7}{N:>6}"
                      f"{t_ref:>9.3f}{t_fus:>9.3f}{t_ref / t_fus:>8.2f}x"
                      f"{r:>10.2e}  {'ok' if ok else 'MISMATCH'}")
        del w

    # The module wrapper, including its fallback: an N the epilogue rejects has
    # to come out right anyway, or "drop-in" is not true.
    layer_ok = True
    # 5056 is deliberately not a multiple of 128, so the middle case has to
    # come out right through the fallback rather than the kernel. "Drop-in" is
    # only true if the shapes the epilogue rejects still work.
    for N, K_local in ((5120, 3072), (5056, 3072), (5120, 3072)):
        w = torch.randn(N, K_local, device=dev, dtype=torch.bfloat16) / 32
        mod = hk_dist.HKRowParallelLinear(w)
        x = torch.randn(64, K_local, device=dev, dtype=torch.bfloat16) / 32
        ref = F.linear(x, w)
        dist.all_reduce(ref)
        r = rel_err(mod(x), ref)
        layer_ok &= r < args.tol
        if rank == 0:
            print(f"\nHKRowParallelLinear N={N}: rel {r:.2e} "
                  f"{'ok' if r < args.tol else 'MISMATCH'}  [{mod.extra_repr()}]")
    ok_all &= layer_ok

    # A 3-D input, because every framework hands these (batch, seq, hidden).
    w = torch.randn(5120, 3072, device=dev, dtype=torch.bfloat16) / 32
    x3 = torch.randn(4, 16, 3072, device=dev, dtype=torch.bfloat16) / 32
    ref = F.linear(x3, w)
    dist.all_reduce(ref)
    got = hk_dist.linear_allreduce(x3, w)
    r3 = rel_err(got, ref)
    ok_all &= got.shape == ref.shape and r3 < args.tol
    if rank == 0:
        print(f"3-D input {tuple(x3.shape)} -> {tuple(got.shape)}: rel {r3:.2e} "
              f"{'ok' if r3 < args.tol else 'MISMATCH'}")

    # The framework patches. vLLM and SGLang are not installed on an RDNA3
    # machine, so nothing here can exercise their forward(); what *can* be
    # exercised is the gate that decides whether to call the kernel at all,
    # which is the part of those files most likely to be wrong. A stand-in
    # layer carrying the attributes both frameworks expose is enough for that.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "integration"))
    import vllm_patch
    import sglang_patch

    class _FakeQuant:
        pass
    _FakeQuant.__name__ = "UnquantizedLinearMethod"

    class _FakeLayer:
        def __init__(self, N, K, **kw):
            self.weight = torch.empty(N, K, device=dev, dtype=torch.bfloat16)
            self.bias = None
            self.tp_size = world
            self.tp_rank = rank
            self.reduce_results = True
            self.input_is_parallel = True
            self.skip_bias_add = False
            self.return_bias = False
            self.quant_method = _FakeQuant()
            self.__dict__.update(kw)

    xb = torch.empty(64, 3072, device=dev, dtype=torch.bfloat16)
    gate_cases = [
        ("supported shape",        _FakeLayer(5120, 3072), xb, True),
        ("N not a multiple of 128", _FakeLayer(5056, 3072), xb, False),
        ("reduce_results off",     _FakeLayer(5120, 3072, reduce_results=False), xb, False),
        ("input not parallel",     _FakeLayer(5120, 3072, input_is_parallel=False), xb, False),
        ("fp16 activations",       _FakeLayer(5120, 3072),
                                   torch.empty(64, 3072, device=dev, dtype=torch.float16), False),
    ]
    gate_ok = True
    for name, layer, xin, want in gate_cases:
        for mod in (vllm_patch, sglang_patch):
            got_gate = mod._eligible(layer, xin)
            if got_gate != want:
                gate_ok = False
                if rank == 0:
                    print(f"  {mod.__name__}._eligible({name}) = {got_gate}, "
                          f"expected {want}  MISMATCH")
    ok_all &= gate_ok
    if rank == 0:
        print(f"framework patch gates ({len(gate_cases)} cases x 2 patches): "
              f"{'ok' if gate_ok else 'MISMATCH'}")

    sync()
    if rank == 0:
        fused = [r for r in rows if r[0] == "prefill"]
        dec = [r for r in rows if r[0] == "decode"]
        if fused:
            print(f"\nprefill speedup {min(r[3] for r in fused):.2f}-"
                  f"{max(r[3] for r in fused):.2f}x, "
                  f"f/g {min(r[4] for r in fused):.2f}-"
                  f"{max(r[4] for r in fused):.2f}")
        if dec:
            print(f"decode  speedup {min(r[3] for r in dec):.2f}-"
                  f"{max(r[3] for r in dec):.2f}x, "
                  f"f/g {min(r[4] for r in dec):.2f}-"
                  f"{max(r[4] for r in dec):.2f}  "
                  f"(against hipBLASLt, so f/g below 1 is our GEMM winning, "
                  f"not the collective vanishing)")
        print("\nALL PASSED" if ok_all else "\nFAILURES ABOVE")

    hk_dist.shutdown()
    dist.destroy_process_group()
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
