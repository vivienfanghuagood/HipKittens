import torch
import tk_kernel
import random
from utils import bench_gemm, print_title

# W7900 (gfx1100) bf16 dense matrix peak, from 96 CU * 2 WMMA/CU/clk * 256
# FLOP/instr at 2.5 GHz boost.
PEAK_TFLOPS = 122.6

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

if __name__ == "__main__":
    gemm_params = {"device": device, "dtype": dtype}

    print_title(f"HipKittens bf16 GEMM (peak {PEAK_TFLOPS} TFLOPS)")
    for shape in bench_shapes:
        gemm_params["shape"] = shape
        bench_gemm(gemm_params, tk_kernel.dispatch_micro, True, num_warmup=20, num_iter=50)

    print_title("PyTorch (hipBLASLt) bf16 GEMM")
    for shape in bench_shapes:
        gemm_params["shape"] = shape
        torch_gemm = lambda A, B, C: torch.matmul(A, B, out=C)
        bench_gemm(gemm_params, torch_gemm, False, num_warmup=20, num_iter=50)
