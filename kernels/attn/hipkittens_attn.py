"""
HipKittens Attention PyTorch Module
Unified wrapper for GQA attention kernels with forward and backward support.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import importlib
import os


class HipKittensAttention(nn.Module):
    """
    Unified PyTorch Module for HipKittens GQA (Grouped Query Attention) kernels.
    
    Supports:
    - Forward-only inference (causal and non-causal)
    - Training with gradients (causal and non-causal)
    - Different input shapes and parameters
    - JIT compilation support
    
    Args:
        causal (bool): Whether to use causal masking
        training (bool): Whether to enable backward pass support
        kernel_path (str): Path to the compiled kernel module
        
    Shape:
        - Q: (batch, seq_len, num_heads, head_dim) - Query tensor
        - K: (batch, seq_len, num_heads_kv, head_dim) - Key tensor  
        - V: (batch, seq_len, num_heads_kv, head_dim) - Value tensor
        - Output: (batch, seq_len, num_heads, head_dim)
        
    Example:
        >>> attn = HipKittensAttention(causal=True, training=False)
        >>> q = torch.randn(16, 2048, 64, 128, dtype=torch.bfloat16, device='cuda')
        >>> k = torch.randn(16, 2048, 8, 128, dtype=torch.bfloat16, device='cuda')
        >>> v = torch.randn(16, 2048, 8, 128, dtype=torch.bfloat16, device='cuda')
        >>> out, lse = attn(q, k, v)
    """
    
    def __init__(
        self,
        causal: bool = False,
        training: bool = False,
        kernel_path: Optional[str] = None,
        return_lse: bool = True
    ):
        super().__init__()
        self.causal = causal
        self.training_mode = training
        self.return_lse = return_lse
        self.kernel_path = kernel_path
        
        # Lazy loading - kernels will be loaded when first used
        self._fwd_kernel = None
        self._bwd_kernel = None
        self._bwd_prep_kernel = None
        
    def _compile_kernel(self, kernel_dir):
        """Auto-compile kernel if not already compiled."""
        import subprocess
        
        # Check if kernel is already compiled by looking for .so files
        so_files = [f for f in os.listdir(kernel_dir) if f.endswith('.so')]
        if so_files:
            return True
        
        print(f"Compiling kernel in {kernel_dir}...")
        try:
            result = subprocess.run(
                ['make', '-C', kernel_dir],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                print(f"✓ Kernel compiled successfully")
                return True
            else:
                print(f"✗ Compilation failed:\n{result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print(f"✗ Compilation timeout")
            return False
        except FileNotFoundError:
            print(f"✗ Make not found. Please install build tools.")
            return False
        except Exception as e:
            print(f"✗ Compilation error: {e}")
            return False
    
    def _load_kernels(self):
        """Lazy load the compiled kernels based on configuration."""
        if self._fwd_kernel is not None:
            return
            
        if self.kernel_path:
            kernel_dir = self.kernel_path
        else:
            # Auto-detect kernel path based on configuration
            base_dir = os.path.dirname(os.path.abspath(__file__))
            if self.training_mode:
                if self.causal:
                    kernel_dir = os.path.join(base_dir, "gqa_causal_backwards")
                else:
                    kernel_dir = os.path.join(base_dir, "gqa_backwards")
            else:
                if self.causal:
                    kernel_dir = os.path.join(base_dir, "gqa_causal")
                else:
                    kernel_dir = os.path.join(base_dir, "gqa")
        
        # Auto-compile if needed
        if not self._compile_kernel(kernel_dir):
            raise RuntimeError(
                f"Failed to compile kernels in {kernel_dir}. "
                f"Please check build environment and dependencies."
            )
        
        # Import kernel module
        try:
            if self.training_mode:
                # Training mode requires forward, backward, and prep kernels
                import sys
                sys.path.insert(0, kernel_dir)
                
                self._fwd_kernel = importlib.import_module("tk_kernel_fwd")
                self._bwd_kernel = importlib.import_module("tk_kernel_bkwd")
                self._bwd_prep_kernel = importlib.import_module("tk_kernel_bkwd_prep")
                
                sys.path.remove(kernel_dir)
            else:
                # Inference mode only needs forward kernel
                import sys
                sys.path.insert(0, kernel_dir)
                
                self._fwd_kernel = importlib.import_module("tk_kernel")
                
                sys.path.remove(kernel_dir)
                
        except ImportError as e:
            raise RuntimeError(
                f"Failed to load kernels from {kernel_dir}. "
                f"Compilation succeeded but import failed. Error: {e}"
            )
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of grouped query attention.
        
        Args:
            q: Query tensor of shape (B, N, H, D)
            k: Key tensor of shape (B, N, H_KV, D)
            v: Value tensor of shape (B, N, H_KV, D)
            
        Returns:
            out: Output tensor of shape (B, N, H, D)
            lse: Log-sum-exp values of shape (B, H, 1, N) if return_lse=True, else None
        """
        # Validate inputs
        self._validate_inputs(q, k, v)
        
        # Load kernels if not already loaded
        self._load_kernels()
        
        # Get dimensions
        B, N, H, D = q.shape
        H_KV = k.shape[2]
        
        if self.training_mode and (q.requires_grad or k.requires_grad or v.requires_grad):
            return HipKittensAttentionFunction.apply(
                q, k, v, self.causal, self.return_lse,
                self._fwd_kernel, self._bwd_kernel, self._bwd_prep_kernel
            )
        else:
            # Inference mode - use simpler forward-only kernel
            return self._forward_inference(q, k, v)
    
    def _forward_inference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for inference (no gradient tracking)."""
        B, N, H, D = q.shape
        H_KV = k.shape[2]
        
        # Allocate output tensors
        out = torch.zeros(B, N, H, D, dtype=q.dtype, device=q.device)
        lse = torch.zeros(B, H, 1, N, dtype=torch.float32, device=q.device)
        
        # Call kernel
        self._fwd_kernel.dispatch_micro(q, k, v, out, lse)
        
        if self.return_lse:
            return out, lse
        else:
            return out, None
    
    def _validate_inputs(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """Validate input tensors."""
        assert q.dim() == 4, f"Q must be 4D (B, N, H, D), got {q.shape}"
        assert k.dim() == 4, f"K must be 4D (B, N, H_KV, D), got {k.shape}"
        assert v.dim() == 4, f"V must be 4D (B, N, H_KV, D), got {v.shape}"
        
        assert q.device == k.device == v.device, "All inputs must be on the same device"
        assert q.device.type == 'cuda', "Inputs must be on CUDA device"
        
        assert q.dtype == k.dtype == v.dtype, "All inputs must have the same dtype"
        assert q.dtype in [torch.bfloat16, torch.float16], "Only bfloat16 and float16 are supported"
        
        B_q, N_q, H_q, D_q = q.shape
        B_k, N_k, H_kv_k, D_k = k.shape
        B_v, N_v, H_kv_v, D_v = v.shape
        
        assert B_q == B_k == B_v, "Batch sizes must match"
        assert N_q == N_k == N_v, "Sequence lengths must match"
        assert D_q == D_k == D_v, "Head dimensions must match"
        assert H_kv_k == H_kv_v, "K and V must have same number of heads"
        assert H_q % H_kv_k == 0, "Query heads must be divisible by KV heads (GQA requirement)"
        
        assert q.is_contiguous(), "Q must be contiguous"
        assert k.is_contiguous(), "K must be contiguous"
        assert v.is_contiguous(), "V must be contiguous"


class HipKittensAttentionFunction(torch.autograd.Function):
    """
    Custom autograd function for HipKittens attention with backward support.
    """
    
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        return_lse: bool,
        fwd_kernel,
        bwd_kernel,
        bwd_prep_kernel,
    ):
        """Forward pass with gradient tracking."""
        B, N, H, D = q.shape
        H_KV = k.shape[2]
        
        # Allocate output tensors
        out = torch.zeros(B, N, H, D, dtype=q.dtype, device=q.device)
        lse = torch.zeros(B, H, N, 1, dtype=torch.float32, device=q.device)
        lse_transposed = lse.transpose(-1, -2).contiguous()
        
        # Call forward kernel
        fwd_kernel.dispatch_fwd(q, k, v, out, lse_transposed)
        
        # Save for backward
        ctx.save_for_backward(q, k, v, out, lse_transposed)
        ctx.causal = causal
        ctx.return_lse = return_lse
        ctx.bwd_kernel = bwd_kernel
        ctx.bwd_prep_kernel = bwd_prep_kernel
        
        if return_lse:
            return out, lse_transposed
        else:
            return out, None
    
    @staticmethod
    def backward(ctx, grad_out, grad_lse):
        """Backward pass."""
        q, k, v, out, lse = ctx.saved_tensors
        
        B, N, H, D = q.shape
        H_KV = k.shape[2]
        
        # Allocate gradient tensors
        dQ_intermediate = torch.zeros(B, H, N, D, dtype=q.dtype, device=q.device)
        dQ = torch.zeros_like(q)
        dK = torch.zeros_like(k)
        dV = torch.zeros_like(v)
        delta = torch.zeros(B, H, N, 1, dtype=torch.float32, device=q.device)
        delta_transposed = delta.transpose(-1, -2).contiguous()
        
        # Prep kernel: compute delta = sum(out * grad_out, dim=-1)
        ctx.bwd_prep_kernel.dispatch_prep(out, grad_out, delta_transposed)
        
        # Backward kernel: compute gradients for Q, K, V
        ctx.bwd_kernel.dispatch_bwd_combined(
            q, k, v,
            grad_out,
            dQ_intermediate,
            dK,
            dV,
            lse,
            delta_transposed
        )
        
        # Shuffle dQ from intermediate format to final format
        ctx.bwd_prep_kernel.dispatch_dq_shuffle(dQ_intermediate, dQ)
        
        return dQ, dK, dV, None, None, None, None, None


class HipKittensAttentionJIT(nn.Module):
    """
    JIT-compilable version of HipKittens Attention.
    
    This version pre-compiles different kernel variants at initialization time
    to support dynamic shapes while maintaining JIT compatibility.
    
    Args:
        causal (bool): Whether to use causal masking
        training (bool): Whether to enable backward pass support
        max_seq_len (int): Maximum sequence length for pre-compilation
        head_dims (list): List of head dimensions to pre-compile
        
    Example:
        >>> attn = HipKittensAttentionJIT(
        ...     causal=True,
        ...     training=False,
        ...     max_seq_len=4096,
        ...     head_dims=[64, 128]
        ... )
        >>> attn = torch.jit.script(attn)
    """
    
    def __init__(
        self,
        causal: bool = False,
        training: bool = False,
        max_seq_len: int = 4096,
        head_dims: list = [64, 128],
    ):
        super().__init__()
        self.causal = causal
        self.training_mode = training
        self.max_seq_len = max_seq_len
        self.head_dims = head_dims
        
        # Pre-compile kernels for each head dimension
        self.kernels = nn.ModuleDict()
        for head_dim in head_dims:
            kernel_key = f"d{head_dim}"
            self.kernels[kernel_key] = HipKittensAttention(
                causal=causal,
                training=training,
                return_lse=True
            )
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with automatic kernel selection based on head dimension.
        
        Args:
            q: Query tensor of shape (B, N, H, D)
            k: Key tensor of shape (B, N, H_KV, D)
            v: Value tensor of shape (B, N, H_KV, D)
            
        Returns:
            out: Output tensor of shape (B, N, H, D)
            lse: Log-sum-exp values of shape (B, H, 1, N)
        """
        D = q.shape[-1]
        
        # Select appropriate kernel based on head dimension
        kernel_key = f"d{D}"
        if kernel_key not in self.kernels:
            raise ValueError(
                f"Head dimension {D} not supported. "
                f"Available dimensions: {self.head_dims}"
            )
        
        return self.kernels[kernel_key](q, k, v)


def create_attention(
    causal: bool = False,
    training: bool = False,
    jit_compile: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create the appropriate attention module.
    
    Args:
        causal: Whether to use causal masking
        training: Whether to enable backward pass support
        jit_compile: Whether to create JIT-compilable version
        **kwargs: Additional arguments passed to the module
        
    Returns:
        Attention module (HipKittensAttention or HipKittensAttentionJIT)
        
    Example:
        >>> # Create causal attention for training
        >>> attn = create_attention(causal=True, training=True)
        >>> 
        >>> # Create JIT-compiled attention for inference
        >>> attn = create_attention(causal=False, jit_compile=True)
        >>> attn = torch.jit.script(attn)
    """
    if jit_compile:
        return HipKittensAttentionJIT(
            causal=causal,
            training=training,
            **kwargs
        )
    else:
        return HipKittensAttention(
            causal=causal,
            training=training,
            **kwargs
        )


__all__ = [
    'HipKittensAttention',
    'HipKittensAttentionFunction',
    'HipKittensAttentionJIT',
    'create_attention',
]
