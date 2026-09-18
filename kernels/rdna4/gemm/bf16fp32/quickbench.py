# One correctness check plus a couple of timings, small enough to sit inside a
# tiling sweep. test.py / bench.py are the real harnesses.
import sys
import torch
import tk_kernel

torch.manual_seed(0)
dtype, device = torch.bfloat16, "cuda:0"

CHECK = (1024, 1024, 512)
SHAPES = [(4096, 4096, 4096), (8192, 8192, 4096)]


def run(m, n, k):
    A = torch.randn((m, k), dtype=dtype, device=device)
    Bt = torch.randn((n, k), dtype=dtype, device=device)
    C = torch.empty((m, n), dtype=dtype, device=device)
    return A, Bt, C


def timed(m, n, k, warmup=10, iters=30):
    A, Bt, C = run(m, n, k)
    for _ in range(warmup):
        tk_kernel.dispatch_micro(A, Bt, C)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        tk_kernel.dispatch_micro(A, Bt, C)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    return 2 * m * n * k / (ms * 1e9)


if __name__ == "__main__":
    # --no-check is for ablation builds, whose results are wrong by construction.
    check = "--no-check" not in sys.argv
    m, n, k = CHECK
    if check:
        A, Bt, C = run(m, n, k)
        tk_kernel.dispatch_micro(A, Bt, C)
        torch.cuda.synchronize()
        ref = torch.matmul(A, Bt.t())
        if not torch.allclose(C, ref, rtol=1e-2, atol=1e-2):
            print("WRONG")
            sys.exit(1)
    print(" ".join(f"{timed(*s):.0f}" for s in SHAPES))
