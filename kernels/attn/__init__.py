"""
HipKittens Attention Module
High-performance Grouped Query Attention for AMD GPUs

This module provides a unified PyTorch interface for HipKittens attention kernels,
supporting various configurations including causal/non-causal, training/inference,
and different GQA configurations.

Quick Start:
    >>> import torch
    >>> from hipkittens_attn import create_attention
    >>> 
    >>> attn = create_attention(causal=True, training=False)
    >>> q = torch.randn(16, 2048, 64, 128, dtype=torch.bfloat16, device='cuda')
    >>> k = torch.randn(16, 2048, 8, 128, dtype=torch.bfloat16, device='cuda')
    >>> v = torch.randn(16, 2048, 8, 128, dtype=torch.bfloat16, device='cuda')
    >>> 
    >>> with torch.no_grad():
    >>>     output, lse = attn(q, k, v)

For detailed documentation, see:
    - README_MODULE.md: Module overview and API reference
    - USAGE_GUIDE.md: Comprehensive examples and patterns
    - QUICK_REFERENCE.md: Quick reference card
    - examples.py: Simple runnable examples
    - test_module.py: Test suite
"""

from .hipkittens_attn import (
    HipKittensAttention,
    HipKittensAttentionFunction,
    HipKittensAttentionJIT,
    create_attention,
)

__version__ = "1.0.0"
__author__ = "HipKittens Team"

__all__ = [
    'HipKittensAttention',
    'HipKittensAttentionFunction', 
    'HipKittensAttentionJIT',
    'create_attention',
]


# Convenience function for version checking
def check_environment():
    """
    Check if the environment is properly configured for using HipKittens attention.
    
    Returns:
        dict: Environment information including PyTorch version, CUDA availability, etc.
    """
    import torch
    
    info = {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device': None,
        'bfloat16_support': False,
    }
    
    if torch.cuda.is_available():
        info['cuda_device'] = torch.cuda.get_device_name(0)
        # Check bfloat16 support
        try:
            x = torch.randn(1, dtype=torch.bfloat16, device='cuda')
            info['bfloat16_support'] = True
        except:
            info['bfloat16_support'] = False
    
    return info


def print_environment_info():
    """Print environment information."""
    info = check_environment()
    print("HipKittens Attention Environment Info:")
    print(f"  PyTorch Version: {info['pytorch_version']}")
    print(f"  CUDA Available: {info['cuda_available']}")
    if info['cuda_device']:
        print(f"  CUDA Device: {info['cuda_device']}")
        print(f"  BFloat16 Support: {info['bfloat16_support']}")
    else:
        print("  ⚠️  WARNING: CUDA not available!")


# Print info when module is imported (can be disabled)
import os
if os.environ.get('HIPKITTENS_VERBOSE', '0') == '1':
    print_environment_info()
