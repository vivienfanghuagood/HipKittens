"""Flash Attention Style Interface Examples - fully compatible with aiter.flash_attn_func"""
import torch
from hipkittens_attn import flash_attn_func, flash_attn_qkvpacked_func, flash_attn_kvpacked_func

print("="*60)
print("HipKittens Flash Attention - Usage Examples")
print("="*60)

# Example 1: Basic usage (fully compatible with aiter.flash_attn_func)
print("\nExample 1: flash_attn_func - Basic Interface")
B, N, H, H_KV, D = 8, 2048, 64, 8, 128
q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device='cuda')
k = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda')
v = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda')

# Inference mode - no requires_grad needed
with torch.no_grad():
    out, lse = flash_attn_func(q, k, v, causal=True, return_lse=True)
print(f"  Inference: {out.shape} ✓")

# Training mode - auto-detect requires_grad
q.requires_grad = True
k.requires_grad = True
v.requires_grad = True
out, lse = flash_attn_func(q, k, v, causal=True, return_lse=True)
loss = out.sum()
loss.backward()
print(f"  Training: {out.shape}, gradients computed ✓")

# Example 2: QKV packed format
print("\nExample 2: flash_attn_qkvpacked_func")
qkv = torch.randn(8, 2048, 3, 64, 128, dtype=torch.bfloat16, device='cuda')
with torch.no_grad():
    out, _ = flash_attn_qkvpacked_func(qkv, causal=True)
print(f"  QKV packed: {out.shape} ✓")

# Example 3: KV packed format (Cross Attention)
print("\nExample 3: flash_attn_kvpacked_func")
q = torch.randn(8, 2048, 64, 128, dtype=torch.bfloat16, device='cuda')
kv = torch.randn(8, 2048, 2, 8, 128, dtype=torch.bfloat16, device='cuda')
with torch.no_grad():
    out, _ = flash_attn_kvpacked_func(q, kv, causal=True)
print(f"  KV packed: {out.shape} ✓")

# Example 4: Integration into nn.Module (like aiter)
print("\nExample 4: Integration into nn.Module (LLaMA style)")
import torch.nn as nn

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

layer = LlamaAttention(4096, 32, 8).cuda().bfloat16()
x = torch.randn(4, 512, 4096, dtype=torch.bfloat16, device='cuda')
out = layer(x)
print(f"  LLaMA layer: input{x.shape} -> output{out.shape} ✓")

print("\n" + "="*60)
print("✅ All interfaces working!")
print("="*60)

