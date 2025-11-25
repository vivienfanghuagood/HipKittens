"""
HipKittens Attention Module
High-performance Grouped Query Attention for AMD GPUs

Provides flash_attn_func compatible API.

Quick Start:
    >>> import torch
    >>> from hipkittens_attn import flash_attn_func
    >>> 
    >>> q = torch.randn(16, 2048, 64, 128, dtype=torch.bfloat16, device='cuda')
    >>> k = torch.randn(16, 2048, 8, 128, dtype=torch.bfloat16, device='cuda')
    >>> v = torch.randn(16, 2048, 8, 128, dtype=torch.bfloat16, device='cuda')
    >>> 
    >>> out, lse = flash_attn_func(q, k, v, causal=True, return_lse=True)
"""

from .hipkittens_attn import (
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_kvpacked_func,
)

__version__ = "1.0.0"
__author__ = "HipKittens Team"

__all__ = [
    'flash_attn_func',
    'flash_attn_qkvpacked_func',
    'flash_attn_kvpacked_func',
]
