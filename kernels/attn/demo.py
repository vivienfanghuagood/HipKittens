"""Simple demo - Flash Attention interface"""
import torch
from hipkittens_attn import flash_attn_func

print("HipKittens Flash Attention - Quick Demo\n")

# Scenario 1: Inference (like aiter)
print("1. Inference Mode")
q = torch.randn(4, 512, 16, 128, dtype=torch.bfloat16, device='cuda')
k = torch.randn(4, 512, 4, 128, dtype=torch.bfloat16, device='cuda')
v = torch.randn(4, 512, 4, 128, dtype=torch.bfloat16, device='cuda')

with torch.no_grad():
    out, lse = flash_attn_func(q, k, v, causal=True, return_lse=True)
print(f"   Output: {out.shape} ✓\n")

# Scenario 2: Training (auto-detect requires_grad)
print("2. Training Mode (auto-detect)")
q.requires_grad = True
k.requires_grad = True
v.requires_grad = True

out, lse = flash_attn_func(q, k, v, causal=True, return_lse=True)
loss = out.sum()
loss.backward()

print(f"   Output: {out.shape}")
print(f"   Q Grad: {q.grad.shape} ✓\n")

print("✅ Flash interface working!")
