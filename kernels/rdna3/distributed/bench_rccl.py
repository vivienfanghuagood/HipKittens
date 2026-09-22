"""What does RCCL actually deliver on this 8x W7900D node?

This is the number every fused compute/communication kernel has to beat, so it
gets measured before anything is written rather than quoted from a spec sheet.
There is no XGMI here -- rocm-smi reports PCIE for all 28 pairs -- and
tools/rdna-probes/p2p_bw.hip measured a single link at 27 GB/s with direct peer
access but only 14.5 GB/s through hipIpcOpenMemHandle, which is the path RCCL
takes. Which of those two RCCL lands on is the question.

No MPI: the pod is air-gapped and has no mpirun. torch.distributed with a
localhost rendezvous spawns the ranks itself, and its "nccl" backend is RCCL
on ROCm.

                                  SAFETY

An earlier version of this script went straight to eight ranks and took the
whole node down -- not the pod, the node. The mechanism is worth writing down,
because nothing about it is specific to this benchmark:

  * Eight torch processes each carry a full ROCm runtime, ~6GB of host RSS.
  * A `medium: Memory` emptyDir is charged to the pod's memory limit, so a
    16Gi /dev/shm was subtracted from the process budget, not added to it.
  * 8 x 6GB + 16Gi against a 64Gi limit is over, so the cgroup OOM killer
    fired -- and it fired while a collective was in flight. The ranks died
    holding queues the driver never got to drain, and the GPUs went with them.

So the failure was not "the benchmark was too big", it was "the benchmark was
killed halfway through a collective". Two guards follow from that, and both
run before any GPU is touched:

  1. fits_in_cgroup() refuses to spawn a world that cannot fit the container's
     memory limit. Dying at argv-parse time is free; dying mid-collective is
     not.
  2. Every collective runs under a watchdog timeout. A hung RCCL op otherwise
     spins forever, and the only way out is the thing that just broke the node.

The default world size is 2, not the device count, so the expensive
configuration has to be asked for by name after the cheap one has been seen to
work. Step up: 2, then 4, then 8.

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
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SIZES_MB = [4, 16, 64]
WARMUP, ITERS = 5, 20

# Host RSS per rank, measured on this image: a bare torch + ROCm context is
# close to 5GB before any allocation. Rounded up, because the consequence of
# guessing low is the failure this guard exists to prevent.
RSS_PER_RANK_GB = 6.5
# A collective that has not finished in this long is hung, not slow. The
# largest size here moves 64MB at a pessimistic 5 GB/s, i.e. tens of
# milliseconds, so this is three orders of magnitude of headroom.
OP_TIMEOUT_S = 60


def cgroup_limit_gb():
    """The container's memory ceiling, or None outside a limited cgroup."""
    for path, parse in (
        ("/sys/fs/cgroup/memory.max", lambda s: s.strip()),                  # v2
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", lambda s: s.strip()),  # v1
    ):
        try:
            with open(path) as f:
                raw = parse(f.read())
        except OSError:
            continue
        if raw == "max":
            return None
        v = int(raw)
        # cgroup v1 reports an absurd sentinel rather than "max".
        return None if v > (1 << 62) else v / 2**30
    return None


def shm_gb():
    """/dev/shm is memory-backed here, so its size is spent, not available."""
    try:
        st = os.statvfs("/dev/shm")
    except OSError:
        return 0.0
    return st.f_blocks * st.f_frsize / 2**30


def fits_in_cgroup(world):
    """Refuse a world size that cannot fit, and say what would."""
    limit = cgroup_limit_gb()
    if limit is None:
        print("no cgroup memory limit found; skipping the budget check")
        return True
    shm = shm_gb()
    need = world * RSS_PER_RANK_GB + shm
    print(f"memory budget: {world} ranks x {RSS_PER_RANK_GB}GB + {shm:.1f}GB "
          f"/dev/shm = {need:.1f}GB against a {limit:.1f}GB limit")
    if need <= limit * 0.85:          # leave the kernel some room to breathe
        return True
    can = int((limit * 0.85 - shm) // RSS_PER_RANK_GB)
    print(f"REFUSING: this OOMs the container, and an OOM during a collective "
          f"takes the node's GPUs down with it.\n"
          f"  raise the pod memory limit to >= {(need / 0.85):.0f}GB, "
          f"or shrink /dev/shm, or run with world <= {max(can, 0)}.",
          file=sys.stderr)
    return False


def bench_one(fn, nbytes, factor):
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
    # Turn a hung collective into a raised exception instead of an infinite
    # spin. Without these, the watchdog timeout below is advisory only.
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

    dist.init_process_group("nccl", rank=rank, world_size=world,
                            timeout=timedelta(seconds=OP_TIMEOUT_S))
    try:
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
                                     2 * f_shard)
            if rank == 0:
                print(f"{'all_reduce':<16}{nbytes:>12}{ms:>10.3f}{alg:>10.1f}{bus:>10.1f}")

            # reduce_scatter: input is the full message, output is 1/n of it.
            # The byte count that matters for bandwidth is the input.
            out = torch.empty(n // world, dtype=torch.bfloat16, device="cuda")
            ms, alg, bus = bench_one(lambda: dist.reduce_scatter_tensor(out, buf),
                                     nbytes, f_shard)
            if rank == 0:
                print(f"{'reduce_scatter':<16}{nbytes:>12}{ms:>10.3f}{alg:>10.1f}{bus:>10.1f}")

            # all_gather: mirror image -- 1/n in, the full message out.
            src = torch.ones(n // world, dtype=torch.bfloat16, device="cuda")
            ms, alg, bus = bench_one(lambda: dist.all_gather_into_tensor(buf, src),
                                     nbytes, f_shard)
            if rank == 0:
                print(f"{'all_gather':<16}{nbytes:>12}{ms:>10.3f}{alg:>10.1f}{bus:>10.1f}")

            del buf, out, src
            torch.cuda.empty_cache()

        dist.barrier()
    finally:
        # Tear the group down even on failure. An abandoned process group is
        # how a dying rank leaves the driver holding work it cannot finish.
        dist.destroy_process_group()


if __name__ == "__main__":
    # Deliberately not device_count(): the large world has to be asked for by
    # name, after the small one has been seen to work.
    world = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    if world > torch.cuda.device_count():
        sys.exit(f"asked for {world} ranks, {torch.cuda.device_count()} GPUs visible")
    if not fits_in_cgroup(world):
        sys.exit(1)
    mp.spawn(worker, args=(world,), nprocs=world, join=True)
