"""bf16 GEMM benchmark, HipKittens against both AMD BLAS libraries.

Three things this harness is careful about, each because getting it wrong
produced a published number that was wrong:

1.  **Both libraries.** torch on ROCm defaults to hipBLASLt, and on gfx1100
    hipBLASLt is the *slower* of the two -- rocBLAS beats it on every shape
    here, by up to 33%. Benchmarking against torch's default alone flatters
    the kernel. `torch.backends.cuda.preferred_blas_library` switches between
    them at runtime ("cublaslt" is hipBLASLt, "cublas" is rocBLAS).

2.  **Every layout.** Tensile tunes per transpose combination and the four are
    not equally covered on this part -- hipBLASLt spans 58..70 TFLOPs across
    NN/NT/TN/TT at 4096^3. HipKittens implements one layout (C = A*B^T, B given
    as (n,k)), so the honest comparison is against the vendor's *best* layout,
    not against whichever one happens to match.

3.  **The denominator.** There is no `PEAK_TFLOPS` constant here any more. The
    122.6 TFLOPs figure is 96 CU x 512 FLOP/clk x 2.495 GHz boost, and this
    card does not run at 2.495 GHz under a GEMM: it sits at 2.0-2.1 GHz against
    its 241 W limit. Percentages are taken against WMMA_CEILING below, which is
    measured rather than derived -- see tools/rdna-probes/wmma_peak.hip.
"""
import random

import torch

import tk_kernel
from utils import time_gemm, print_title

# Back-to-back v_wmma_f32_16x16x16_bf16 with no memory access at all, measured
# by tools/rdna-probes/wmma_peak.hip on this card. Flat from 2 to 16 independent
# accumulator chains, so it is an issue-rate ceiling and not a latency artifact.
WMMA_CEILING = 100.5

bench_shapes = [
    (4096, 4096, 4096), # (m, n, k)
    (8192, 8192, 4096),
    (4096, 8192, 2048),
    (8192, 4096, 2048),
    (2048, 4096, 4096),
    (2048, 2048, 4096),
    (2048, 2048, 2048),
]

torch.manual_seed(0)
random.seed(0)
dtype = torch.bfloat16
device = "cuda:0"


def vendor_best(shape, backend):
    """Best TFLOPs over all four transpose combinations, for one BLAS backend."""
    m, n, k = shape
    torch.backends.cuda.preferred_blas_library(backend)
    A  = torch.randn((m, k), dtype=dtype, device=device)
    At = torch.randn((k, m), dtype=dtype, device=device)
    B  = torch.randn((k, n), dtype=dtype, device=device)
    Bt = torch.randn((n, k), dtype=dtype, device=device)
    C  = torch.empty((m, n), dtype=dtype, device=device)
    layouts = {
        "NN": lambda: torch.matmul(A,      B,      out=C),
        "NT": lambda: torch.matmul(A,      Bt.t(), out=C),
        "TN": lambda: torch.matmul(At.t(), B,      out=C),
        "TT": lambda: torch.matmul(At.t(), Bt.t(), out=C),
    }
    best_tf, best_lay = 0.0, ""
    for lay, fn in layouts.items():
        for _ in range(15):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(50):
            fn()
        e.record()
        torch.cuda.synchronize()
        tf = 2*m*n*k / ((s.elapsed_time(e) / 50) * 1e9)
        if tf > best_tf:
            best_tf, best_lay = tf, lay
    del A, At, B, Bt, C
    torch.cuda.empty_cache()
    return best_tf, best_lay


if __name__ == "__main__":
    print_title(f"bf16 GEMM on gfx1100 -- WMMA ceiling {WMMA_CEILING} TFLOPs", 76)
    print(f"{'shape (MxNxK)':>18} | {'hipBLASLt':>13} | {'rocBLAS':>13} | "
          f"{'HipKittens':>10} | {'HK/best':>7} | {'HK/ceil':>7}")
    print("-" * 82)

    for shape in bench_shapes:
        m, n, k = shape
        hipblaslt = vendor_best(shape, "cublaslt")
        rocblas   = vendor_best(shape, "cublas")

        params = {"device": device, "dtype": dtype, "shape": shape}
        hk = 2*m*n*k / (time_gemm(params, tk_kernel.dispatch_micro, True) * 1e9)

        best = max(hipblaslt[0], rocblas[0])
        print(f"{m}x{n}x{k:<6} | {hipblaslt[0]:8.1f} ({hipblaslt[1]}) | "
              f"{rocblas[0]:8.1f} ({rocblas[1]}) | {hk:10.1f} | "
              f"{100*hk/best:6.0f}% | {100*hk/WMMA_CEILING:6.0f}%")
