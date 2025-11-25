"""最简单的演示 - 验证自动编译和基本功能"""
import torch

print("HipKittens Attention - 快速演示\n")

# 导入（会在首次forward时自动编译）
from hipkittens_attn import create_attention

# 场景1: 推理
print("1. 推理模式")
attn = create_attention(causal=True, training=False)
q = torch.randn(4, 512, 16, 128, dtype=torch.bfloat16, device='cuda')
k = torch.randn(4, 512, 4, 128, dtype=torch.bfloat16, device='cuda')
v = torch.randn(4, 512, 4, 128, dtype=torch.bfloat16, device='cuda')

with torch.no_grad():
    output, lse = attn(q, k, v)
print(f"   输出: {output.shape} ✓\n")

# 场景2: 训练
print("2. 训练模式（带梯度）")
attn_train = create_attention(causal=True, training=True)
q = torch.randn(4, 512, 16, 128, dtype=torch.bfloat16, device='cuda', requires_grad=True)
k = torch.randn(4, 512, 4, 128, dtype=torch.bfloat16, device='cuda', requires_grad=True)
v = torch.randn(4, 512, 4, 128, dtype=torch.bfloat16, device='cuda', requires_grad=True)

output, lse = attn_train(q, k, v)
loss = output.sum()
loss.backward()

print(f"   输出: {output.shape}")
print(f"   Q梯度: {q.grad.shape} ✓\n")

print("✓ 所有功能正常！")
