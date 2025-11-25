# HipKittens Attention

Flash Attention style interface, compatible with `aiter.flash_attn_func`, powered by HipKittens GQA kernels.

## Quick Start

```python
import torch
from hipkittens_attn import flash_attn_func

# Prepare inputs (BNHD format)
q = torch.randn(16, 2048, 64, 128, dtype=torch.bfloat16, device='cuda')
k = torch.randn(16, 2048, 8, 128, dtype=torch.bfloat16, device='cuda')
v = torch.randn(16, 2048, 8, 128, dtype=torch.bfloat16, device='cuda')

# Inference - auto-compile kernel
with torch.no_grad():
    out, lse = flash_attn_func(q, k, v, causal=True, return_lse=True)

# Training - auto-detect requires_grad to enable backward
q.requires_grad = True
out, lse = flash_attn_func(q, k, v, causal=True)
loss.backward()
```

## API

### flash_attn_func

```python
out, lse = flash_attn_func(
    q, k, v,              # (B,N,H,D), (B,N,H_KV,D), (B,N,H_KV,D)
    dropout_p=0.0,        # not supported
    softmax_scale=None,   # not supported
    causal=False,         # causal masking
    window_size=(-1,-1),  # not supported
    alibi_slopes=None,    # not supported
    deterministic=False,  # not supported
    return_lse=False,     # return LSE
)
```

### flash_attn_qkvpacked_func

```python
qkv = torch.randn(B, N, 3, H, D, ...)  # QKV packed
out, lse = flash_attn_qkvpacked_func(qkv, causal=True)
```

### flash_attn_kvpacked_func

```python
kv = torch.randn(B, N, 2, H_KV, D, ...)  # KV packed
out, lse = flash_attn_kvpacked_func(q, kv, causal=True)
```

## Usage Examples

### 1. LLaMA Inference

```python
from hipkittens_attn import flash_attn_func

q = torch.randn(16, 4096, 32, 128, dtype=torch.bfloat16, device='cuda')
k = torch.randn(16, 4096, 8, 128, dtype=torch.bfloat16, device='cuda')
v = torch.randn(16, 4096, 8, 128, dtype=torch.bfloat16, device='cuda')

with torch.no_grad():
    out, _ = flash_attn_func(q, k, v, causal=True)
```

### 2. LLaMA Training

```python
q.requires_grad = k.requires_grad = v.requires_grad = True
out, lse = flash_attn_func(q, k, v, causal=True, return_lse=True)
loss = out.sum()
loss.backward()  # gradients in q.grad, k.grad, v.grad
```

### 3. Integration into nn.Module (like aiter)

```python
import torch.nn as nn
from hipkittens_attn import flash_attn_func

class LlamaAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
    
    def forward(self, x):
        B, N, _ = x.shape
        q = self.wq(x).view(B, N, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, N, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, N, self.n_kv_heads, self.head_dim)
        
        # Use flash_attn_func, just like aiter
        out, _ = flash_attn_func(q, k, v, causal=True)
        
        return self.wo(out.flatten(-2))
```

## Input/Output

**Inputs (BNHD):**
- Q: `(Batch, SeqLen, Heads, HeadDim)`
- K: `(Batch, SeqLen, KV_Heads, HeadDim)`
- V: `(Batch, SeqLen, KV_Heads, HeadDim)`

**Outputs:**
- out: `(Batch, SeqLen, Heads, HeadDim)`
- lse: `(Batch, Heads, 1, SeqLen)` or None

**Requirements:**
- dtype: `torch.bfloat16` or `torch.float16`
- device: `cuda`
- layout: contiguous
- GQA: `H % H_KV == 0`

## Automation Features

- ✅ **Auto-compile** - First use auto-executes make to compile kernels
- ✅ **Auto-mode** - Auto-selects inference/training based on `requires_grad`
- ✅ **Auto-select** - Auto-selects correct kernel based on causal parameter

## Kernel Selection

| causal | requires_grad | Kernel Directory |
|--------|---------------|------------------|
| False | False | `gqa/` |
| True | False | `gqa_causal/` |
| False | True | `gqa_backwards/` |
| True | True | `gqa_causal_backwards/` |

## Testing

```bash
python example_flash.py  # Flash interface examples
python demo.py           # Simple demo
```

## Comparison with aiter

```python
# aiter
import aiter
out, lse = aiter.flash_attn_func(q, k, v, causal=True, return_lse=True)

# hipkittens (fully compatible)
import hipkittens_attn
out, lse = hipkittens_attn.flash_attn_func(q, k, v, causal=True, return_lse=True)

# or direct import
from hipkittens_attn import flash_attn_func
out, lse = flash_attn_func(q, k, v, causal=True, return_lse=True)
```

## GQA Configurations

```python
# MHA: H_KV = H
q, k, v = (B,N,32,D), (B,N,32,D), (B,N,32,D)

# GQA: H % H_KV = 0
q, k, v = (B,N,64,D), (B,N,8,D), (B,N,8,D)

# MQA: H_KV = 1
q, k, v = (B,N,32,D), (B,N,1,D), (B,N,1,D)
```
