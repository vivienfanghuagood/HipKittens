"""What does RCCL actually deliver on this 8x W7900D node?

This is the number every fused compute/communication kernel has to beat, so it
gets measured before anything is written rather than quoted from a spec sheet.
There is no XGMI here -- rocm-smi reports PCIE for all 28 pairs -- and
tools/rdna-probes/p2p_bw.hip measured a single link at 27 GB/s with direct peer
access but only 14.5 GB/s through hipIpcOpenMemHandle, which is the path RCCL
takes. Which of those two RCCL lands on is the question.

No MPI: the pod is air-gapped and has no mpirun. torch.distributed with a
localhost rendezvous spawns the eight ranks itself, and its "nccl" backend is
RCCL on ROCm.

Reported the way nccl-tests does it, because the two numbers answer different
questions:

  algbw = bytes / time             what the caller experiences
  busbw = algbw * factor           what the wire carries, so it can be compared
                                   against the 27 / 14.5 GB/s point-to-point
                                   measurements directly

The factor is the standard one per collective: (n-1)/n for reduce_scatter and
all_gather, which each move every byte past n-1 hops, and 2(n-1)/n for
all_reduce, which is a reduce_scatter followed by an all_gather.
"""
import os, sys, time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SIZES_MB = [4, 16, 64, 256]
WARMUP, ITERS = 5, 20


def bench_one(fn, nbytes, factor, world):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / ITERS
    algbw = nbytes / dt / 1e9
    return dt * 1e3, algbw, algbw * factor


def worker(rank, world):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)

    if rank == 0:
        print(f"RCCL on {world} x {torch.cuda.get_device_name(0)}")
        print(f"{'collective':<16}{'bytes':>12}{'ms':>10}{'algbw':>10}{'busbw':>10}")

    f_shard = (world - 1) / world
    for mb in SIZES_MB:
        nbytes = mb << 20
        n = nbytes // 2  # bf16 elements, total message size

        # all_reduce: input and output are both the full message.
        buf = torch.ones(n, dtype=torch.bfloat16, device="cuda")
        ms, alg, bus = bench_one(lambda: dist.all_reduce(buf), nbytes,
                                 2 * f_shard, world)
        if rank == 0:
            print(f"{'all_reduce':<16}{nbytes:>12}{ms:>10.3f}{alg:>10.1f}{bus:>10.1f}")

        # reduce_scatter: input is the full message, output is 1/n of it. The
        # byte count that matters for bandwidth is the input.
        out = torch.empty(n // world, dtype=torch.bfloat16, device="cuda")
        ms, alg, bus = bench_one(lambda: dist.reduce_scatter_tensor(out, buf),
                                 nbytes, f_shard, world)
        if rank == 0:
            print(f"{'reduce_scatter':<16}{nbytes:>12}{ms:>10.3f}{alg:>10.1f}{bus:>10.1f}")

        # all_gather: mirror image -- 1/n in, the full message out.
        src = torch.ones(n // world, dtype=torch.bfloat16, device="cuda")
        ms, alg, bus = bench_one(lambda: dist.all_gather_into_tensor(buf, src),
                                 nbytes, f_shard, world)
        if rank == 0:
            print(f"{'all_gather':<16}{nbytes:>12}{ms:>10.3f}{alg:>10.1f}{bus:>10.1f}")

        del buf, out, src
        torch.cuda.empty_cache()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    world = int(sys.argv[1]) if len(sys.argv) > 1 else torch.cuda.device_count()
    mp.spawn(worker, args=(world,), nprocs=world, join=True)
