#!/usr/bin/env python3
"""The path vLLM/SGLang actually take, on the same shapes as gemm_rs.

gemm_rs measures itself against *its own* GEMM plus a hipMemcpyPeerAsync. That
is the right control for "did fusing help", and the wrong control for "is this
worth deploying". A framework does not call my GEMM: it calls hipBLASLt, and it
does not memcpy, it calls RCCL. Either of those could be faster than what the
fused kernel is being compared to, in which case the 1.4x is against a
strawman.

So this measures, on the identical shapes:

  gemm_ms   F.linear, i.e. hipBLASLt (or rocBLAS, whichever the heuristic
            picks) at the exact layout a row-parallel Linear produces:
            A[M, K_local] row-major, W[N, K_local] row-major, out = A @ W.T.
  ar_ms     dist.all_reduce -- what plain TP does after a row-parallel layer.
  rs_ms     dist.reduce_scatter_tensor -- what TP + sequence parallelism does,
            and the collective gemm_rs actually implements.
  total     gemm + the collective, serialised, because that is the framework
            path: the collective cannot start until the GEMM has finished.

all_reduce is reported alongside reduce_scatter because it is the *default*.
Sequence parallelism has to be switched on to get reduce_scatter, and for
decode at M=1 it cannot be switched on at all -- one row does not split across
two ranks -- so all_reduce is the only number decode has.

Timing is back-to-back enqueue with a single sync at the end, not a sync per
iteration. That is deliberately the generous reading for the baseline: it gives
the framework path free pipelining across iterations that gemm_rs's own harness
does not give itself.

Run:  HIP_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 baseline_torch.py
      --tune      let TunableOp sweep solutions per shape first, which is the
                  "rocBLAS best" number rather than the heuristic's guess.
      --rotate N  cycle N weight replicas so the weights cannot sit in cache;
                  see the comment on Ws below -- this matters for decode.
      --decode    skip the prefill half.
"""
import argparse
import datetime
import os
import time

import torch
import torch.distributed as dist

# Qwen3.8-27B. hidden 5120, intermediate 17408, attention output width 6144
# (24 heads x 256 for the full-attention layers, 48 x 128 for the linear ones).
H, I, ATTN = 5120, 17408, 6144
CONC = [1, 2, 4, 8, 16, 32]
CHUNK = 8192  # prefill token cap per forward


def bench(fn, iters):
    for _ in range(5):
        fn()
    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    dist.barrier()
    return (t1 - t0) * 1e3 / iters


def run_shape(rank, world, phase, conc, op, M, K, rotate=1):
    K_local = K // world
    dev = torch.cuda.current_device()
    # A row-parallel Linear: the weight is sharded down K, every rank holds the
    # full N. randn and not zeros -- a denormal-free input, and the timing of a
    # GEMM should never depend on its data, but proving that is not this
    # script's job and constant inputs invite a compiler to help.
    A = torch.randn(M, K_local, device=dev, dtype=torch.bfloat16)
    C = torch.empty(M, H, device=dev, dtype=torch.bfloat16)

    # `rotate` weight replicas, cycled one per iteration.
    #
    # This exists because the decode GEMM measured 938 GB/s on an 864 GB/s part.
    # A decode GEMM is pure weight streaming, so a number above the memory peak
    # is not a fast kernel, it is a cache hit: mlp_down's shard is 89 MB and
    # Navi 31's Infinity Cache is 96 MB, so re-running the same GEMM 200 times
    # leaves the weight resident and measures a working set no real decode loop
    # has -- 64 layers of distinct weights stream past once per token, and none
    # of them is still there when it is next needed. rotate=8 puts ~700 MB in
    # play, which the cache cannot hold.
    Ws = [torch.randn(H, K_local, device=dev, dtype=torch.bfloat16)
          for _ in range(rotate)]
    step = {"i": 0}

    flops = 2.0 * M * K_local * H
    iters = int(max(5, min(200, 2e12 / max(flops, 1.0))))

    def gemm():
        W = Ws[step["i"]]
        step["i"] = (step["i"] + 1) % rotate
        torch.matmul(A, W.t(), out=C)

    gemm_ms = bench(gemm, iters)

    def allreduce():
        dist.all_reduce(C)

    ar_ms = bench(allreduce, iters)

    def gemm_ar():
        gemm()
        dist.all_reduce(C)

    ar_total = bench(gemm_ar, iters)

    # reduce_scatter needs M divisible by the world size. At M=1 it is not, and
    # no amount of configuration makes it so -- that is a real property of
    # decode at concurrency 1, not a gap in this script.
    if M % world == 0:
        S = torch.empty(M // world, H, device=dev, dtype=torch.bfloat16)

        def rs():
            dist.reduce_scatter_tensor(S, C)

        rs_ms = bench(rs, iters)

        def gemm_rs_():
            gemm()
            dist.reduce_scatter_tensor(S, C)

        rs_total = bench(gemm_rs_, iters)
    else:
        rs_ms = rs_total = float("nan")

    if rank == 0:
        tf = flops / (gemm_ms * 1e-3) / 1e12
        # Weight bytes moved per GEMM. For decode this is the whole story and
        # the ratio to the part's 864 GB/s peak is the number to read, not
        # TFLOPs -- a decode GEMM is weight streaming, not arithmetic.
        gbs = H * K_local * 2 / (gemm_ms * 1e-3) / 1e9
        print(
            f"{phase:<8} {conc:4d} {op:<9} {M:6d} {K:7d} {K_local:6d} {H:5d} "
            f"{gemm_ms:8.3f} {tf:7.1f} {gbs:7.0f} {ar_ms:8.3f} {ar_total:8.3f} "
            f"{rs_ms:8.3f} {rs_total:8.3f}",
            flush=True,
        )

    del A, C, Ws
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true",
                    help="autotune each GEMM shape with TunableOp first")
    ap.add_argument("--rotate", type=int, default=1,
                    help="number of weight replicas to cycle (defeats cache)")
    ap.add_argument("--decode", action="store_true", help="decode shapes only")
    args = ap.parse_args()

    if args.tune:
        # Has to be set before the first GEMM. TunableOp sweeps the available
        # hipBLASLt/rocBLAS solutions for each (shape, layout, dtype) and keeps
        # the fastest, which is the number a vendor library can actually reach
        # rather than what its shipped heuristic guesses.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        torch.cuda.tunable.set_filename("tunableop_qwen_tp2.csv")

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # A bounded timeout, not the 30 minute default: if RCCL wedges on this
    # fabric the useful outcome is an error, not a process that has to be found
    # and killed later.
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=180))

    if rank == 0:
        name = torch.cuda.get_device_name()
        print(f"=== torch {torch.__version__} / RCCL baseline, {world} x {name} ===")
        print(f"tunableop: {'on' if args.tune else 'off (shipped heuristic)'}"
              f"   weight replicas: {args.rotate}")
        print("\nThe framework path: hipBLASLt GEMM, then a collective. ar_* is "
              "plain TP\n(all_reduce); rs_* is TP + sequence parallelism "
              "(reduce_scatter), which is\nwhat gemm_rs implements. *_total is "
              "GEMM and collective serialised.\n")
        print(f"{'phase':<8} {'conc':>4} {'op':<9} {'M':>6} {'K':>7} "
              f"{'K_loc':>6} {'N':>5} {'gemm_ms':>8} {'TFLOPs':>7} "
              f"{'W_GB/s':>7} {'ar_ms':>8} {'ar_total':>8} {'rs_ms':>8} "
              f"{'rs_total':>8}",
              flush=True)

    if not args.decode:
        for c in CONC:
            M = min(c * 2048, CHUNK)
            run_shape(rank, world, "prefill", c, "attn_out", M, ATTN, args.rotate)
            run_shape(rank, world, "prefill", c, "mlp_down", M, I, args.rotate)
    for c in CONC:
        run_shape(rank, world, "decode", c, "attn_out", c, ATTN, args.rotate)
        run_shape(rank, world, "decode", c, "mlp_down", c, I, args.rotate)

    if args.tune and rank == 0:
        torch.cuda.tunable.write_file()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
