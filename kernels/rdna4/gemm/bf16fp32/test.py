import torch
import tk_kernel
import random
from utils import init_randn, init_empty

# Shapes must be multiples of the block tile: 128 (m), 128 (n), 32 (k).
test_shapes = [
    (4096, 4096, 4096), # (m, n, k)
    (2048, 4096, 2048),
    (4096, 2048, 2048),
    (4096, 1024, 4096),
    (512, 1024, 1024),
    (512, 1024, 2048),
    (2048, 1024, 512),
    (1024, 1024, 128),
]

torch.manual_seed(0)
random.seed(0)
dtype = torch.bfloat16
device = "cuda:0"

if __name__ == "__main__":
    for test_shape in test_shapes:
        m, n, k = test_shape
        A = init_randn((m, k), dtype, device)
        B = init_randn((k, n), dtype, device)
        Bt = B.t().contiguous()
        C = init_empty((m, n), dtype, device)

        C_ref = torch.matmul(A, B)
        tk_kernel.dispatch_micro(A, Bt, C)
        torch.cuda.synchronize()

        is_valid = torch.allclose(C, C_ref, rtol=1e-2, atol=1e-2)
        result = "TEST PASSED" if is_valid else "TEST FAILED"
        if not is_valid:
            err = (C.float() - C_ref.float()).abs().max().item()
            result += f" (max abs err {err:.4g})"
        print(f"{test_shape}".ljust(20) + f" | {result}")
