"""
HipKittens Attention - Flash Attention Style Interface
Provides flash_attn_func compatible API for HipKittens GQA kernels.
"""

import torch
from typing import Optional, Tuple
import importlib
import os
import subprocess
import glob


# ============================================================================
# Kernel Management
# ============================================================================

class _KernelLoader:
    """Manages kernel compilation and loading."""
    
    def __init__(self):
        self.cache = {}
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
    
    def _get_kernel_dir(self, causal: bool, training: bool, n: int) -> Tuple[str, str]:
        """Get kernel source directory and JIT build directory."""
        if training:
            base = "gqa_causal_backwards" if causal else "gqa_backwards"
        else:
            base = "gqa_causal" if causal else "gqa"
        
        kernel_dir = os.path.join(self.base_dir, base)
        jit_dir = os.path.join(kernel_dir, f"build_N{n}")
        os.makedirs(jit_dir, exist_ok=True)
        
        return kernel_dir, jit_dir
    
    def _check_compiled(self, jit_dir: str, training: bool) -> bool:
        """Check if required kernels are compiled."""
        if training:
            required = ['tk_kernel_fwd', 'tk_kernel_bkwd', 'tk_kernel_bkwd_prep']
        else:
            required = ['tk_kernel']
        
        for name in required:
            if not glob.glob(os.path.join(jit_dir, f"{name}*.so")):
                return False
        return True
    
    def _compile(self, kernel_dir: str, jit_dir: str, n: int) -> bool:
        """JIT compile kernels for specific sequence length."""
        # Check if THUNDERKITTENS_ROOT is set
        if 'THUNDERKITTENS_ROOT' not in os.environ:
            print("ERROR: THUNDERKITTENS_ROOT environment variable not set")
            print("Please run: source env.src")
            return False
        
        print(f"JIT compiling kernels for N={n}... (this may take 1-2 minutes)")
        
        try:
            result = subprocess.run(
                ['make', '-C', kernel_dir, f'ATTN_N={n}'],
                capture_output=True,
                text=True,
                timeout=600,
                env=os.environ.copy()
            )
            
            if result.returncode != 0:
                print(f"Kernel compilation failed for N={n}")
                print(f"STDOUT:\n{result.stdout}")
                print(f"STDERR:\n{result.stderr}")
                return False
            
            # Move compiled .so files to jit_dir
            for so_file in glob.glob(os.path.join(kernel_dir, "*.so")):
                basename = os.path.basename(so_file)
                dest = os.path.join(jit_dir, basename)
                import shutil
                shutil.move(so_file, dest)
            
            print(f"✓ Successfully compiled kernels for N={n}")
            return True
            
        except Exception as e:
            print(f"Exception during compilation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_modules(self, jit_dir: str, training: bool):
        """Load compiled kernel modules."""
        import sys
        sys.path.insert(0, jit_dir)
        try:
            if training:
                fwd = importlib.import_module("tk_kernel_fwd")
                bwd = importlib.import_module("tk_kernel_bkwd")
                prep = importlib.import_module("tk_kernel_bkwd_prep")
                return fwd, bwd, prep
            else:
                fwd = importlib.import_module("tk_kernel")
                return fwd, None, None
        finally:
            sys.path.remove(jit_dir)
    
    def get_kernels(self, causal: bool, training: bool, n: int):
        """Get or JIT compile kernels for configuration."""
        key = (causal, training, n)
        if key in self.cache:
            return self.cache[key]
        
        kernel_dir, jit_dir = self._get_kernel_dir(causal, training, n)
        
        # Auto-compile if needed (JIT compilation)
        if not self._check_compiled(jit_dir, training):
            if not self._compile(kernel_dir, jit_dir, n):
                raise RuntimeError(f"Failed to JIT compile kernels for N={n}")
        
        # Load modules
        kernels = self._load_modules(jit_dir, training)
        self.cache[key] = kernels
        return kernels


_KERNEL_LOADER = _KernelLoader()


# ============================================================================
# Autograd Function
# ============================================================================

class _FlashAttnFunc(torch.autograd.Function):
    """Autograd function for attention with backward support."""
    
    @staticmethod
    def forward(ctx, q, k, v, causal, return_lse, fwd_kernel, bwd_kernel, prep_kernel):
        B, N, H, D = q.shape
        
        # Allocate outputs
        out = torch.zeros(B, N, H, D, dtype=q.dtype, device=q.device)
        lse = torch.zeros(B, H, 1, N, dtype=torch.float32, device=q.device)
        
        # Forward
        fwd_kernel.dispatch_fwd(q, k, v, out, lse)
        
        # Save for backward
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.bwd_kernel = bwd_kernel
        ctx.prep_kernel = prep_kernel
        
        return out, lse if return_lse else None
    
    @staticmethod
    def backward(ctx, grad_out, grad_lse):
        q, k, v, out, lse = ctx.saved_tensors
        B, N, H, D = q.shape
        
        # Allocate gradients
        dQ_intermediate = torch.zeros(B, H, N, D, dtype=q.dtype, device=q.device)
        dQ = torch.zeros_like(q)
        dK = torch.zeros_like(k)
        dV = torch.zeros_like(v)
        delta = torch.zeros(B, H, 1, N, dtype=torch.float32, device=q.device)
        
        # Backward prep
        ctx.prep_kernel.dispatch_prep(out, grad_out, delta)
        
        # Backward combined
        ctx.bwd_kernel.dispatch_bwd_combined(
            q, k, v, grad_out,
            dQ_intermediate, dK, dV,
            lse, delta
        )
        
        # Shuffle dQ
        ctx.prep_kernel.dispatch_dq_shuffle(dQ_intermediate, dQ)
        
        return dQ, dK, dV, None, None, None, None, None


# ============================================================================
# Main Interface - Flash Attention Style
# ============================================================================

def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_lse: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Flash Attention function - compatible with aiter.flash_attn_func API.
    
    Args:
        q: Query (B, N, H, D)
        k: Key (B, N, H_KV, D)
        v: Value (B, N, H_KV, D)
        dropout_p: Dropout (not supported, must be 0.0)
        softmax_scale: Softmax scale (not supported)
        causal: Causal masking
        window_size: Window size (not supported)
        alibi_slopes: ALiBi (not supported)
        deterministic: Deterministic mode (not supported)
        return_lse: Return log-sum-exp
        
    Returns:
        out: (B, N, H, D)
        lse: (B, H, 1, N) if return_lse else None
        
    Example:
        >>> out, lse = flash_attn_func(q, k, v, causal=True, return_lse=True)
    """
    # Validate inputs
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Inputs must be 4D (B,N,H,D)"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be on CUDA"
    assert q.dtype in [torch.bfloat16, torch.float16], "Only bfloat16/float16 supported"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "Inputs must be contiguous"
    
    B, N, H, D = q.shape
    assert k.shape[:2] == (B, N) and v.shape[:2] == (B, N), "Batch and seq_len must match"
    assert k.shape[2] == v.shape[2], "K and V must have same num_heads"
    assert k.shape[3] == D and v.shape[3] == D, "Head dim must match"
    assert H % k.shape[2] == 0, "H must be divisible by H_KV (GQA)"
    
    # Validate unsupported params
    if dropout_p != 0.0:
        raise NotImplementedError("dropout_p not supported")
    if softmax_scale is not None:
        raise NotImplementedError("softmax_scale not supported")
    if window_size != (-1, -1):
        raise NotImplementedError("window_size not supported")
    if alibi_slopes is not None:
        raise NotImplementedError("alibi_slopes not supported")
    if deterministic:
        raise NotImplementedError("deterministic not supported")
    
    # Auto-detect training mode
    training = q.requires_grad or k.requires_grad or v.requires_grad
    
    # Get kernels (JIT compile if needed for this N)
    fwd_kernel, bwd_kernel, prep_kernel = _KERNEL_LOADER.get_kernels(causal, training, N)
    
    # Forward-only path
    if not training:
        out = torch.zeros(B, N, H, D, dtype=q.dtype, device=q.device)
        lse = torch.zeros(B, H, 1, N, dtype=torch.float32, device=q.device)
        
        fwd_kernel.dispatch_micro(q, k, v, out, lse)
        
        return out, (lse if return_lse else None)
    
    # Training path with autograd
    return _FlashAttnFunc.apply(q, k, v, causal, return_lse, fwd_kernel, bwd_kernel, prep_kernel)


def flash_attn_qkvpacked_func(
    qkv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_lse: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Flash Attention with packed QKV.
    
    Args:
        qkv: (B, N, 3, H, D)
        
    Returns:
        out: (B, N, H, D)
        lse: (B, H, 1, N) if return_lse else None
    """
    assert qkv.dim() == 5 and qkv.shape[2] == 3, "qkv must be (B,N,3,H,D)"
    q, k, v = qkv.unbind(dim=2)
    return flash_attn_func(q, k, v, dropout_p, softmax_scale, causal, 
                           window_size, alibi_slopes, deterministic, return_lse)


def flash_attn_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_lse: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Flash Attention with packed KV.
    
    Args:
        q: (B, N, H, D)
        kv: (B, N, 2, H_KV, D)
        
    Returns:
        out: (B, N, H, D)
        lse: (B, H, 1, N) if return_lse else None
    """
    assert kv.dim() == 5 and kv.shape[2] == 2, "kv must be (B,N,2,H_KV,D)"
    k, v = kv.unbind(dim=2)
    return flash_attn_func(q, k, v, dropout_p, softmax_scale, causal,
                           window_size, alibi_slopes, deterministic, return_lse)


__all__ = [
    'flash_attn_func',
    'flash_attn_qkvpacked_func',
    'flash_attn_kvpacked_func',
]
