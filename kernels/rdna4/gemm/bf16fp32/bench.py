import torch
import tk_kernel
import random
from utils import bench_gemm, print_title

# RX 9070 XT (gfx1201, Navi 48) bf16 dense matrix peak: 64 CU * 1024
# FLOP/CU/clk at 2.97 GHz boost = 194.6 TFLOPS, which is AMD's published figure.
# RDNA4 doubled matrix throughput per CU over RDNA3's 512 FLOP/CU/clk.
#
# Change this if you are on a different part -- CUs * 1024 * boost_GHz / 1000.
# AMD publishes 145 for the RX 9070 (56 CU) and 137 for the 9070 GRE (48 CU).
PEAK_TFLOPS = 194.6

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
