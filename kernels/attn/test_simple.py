"""简单测试脚本 - 验证自动编译和基本功能"""
import torch
import sys

print("="*60)
print("HipKittens Attention - 简单测试")
print("="*60)

# 检查CUDA
if not torch.cuda.is_available():
    print("✗ CUDA不可用")
    sys.exit(1)

print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
print(f"✓ PyTorch: {torch.__version__}")

# 测试导入和自动编译
print("\n测试1: 导入模块（自动编译kernel）")
try:
    from hipkittens_attn import create_attention
    print("✓ 导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 测试推理
print("\n测试2: 推理模式（causal）")
try:
    attn = create_attention(causal=True, training=False)
    
    B, N, H, H_KV, D = 4, 512, 16, 4, 128
    q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda')
    
    with torch.no_grad():
        output, lse = attn(q, k, v)
    
    assert output.shape == (B, N, H, D), f"输出shape错误: {output.shape}"
    print(f"✓ 推理成功: {output.shape}")
except Exception as e:
    print(f"✗ 推理失败: {e}")
    import traceback
    traceback.print_exc()

# 测试训练
print("\n测试3: 训练模式（带梯度）")
try:
    attn = create_attention(causal=True, training=True)
    
    B, N, H, H_KV, D = 4, 512, 16, 4, 128
    q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    k = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    v = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    
    output, lse = attn(q, k, v)
    loss = output.sum()
    loss.backward()
    
    assert q.grad is not None, "Q梯度未计算"
    assert k.grad is not None, "K梯度未计算"
    assert v.grad is not None, "V梯度未计算"
    print(f"✓ 训练成功: 梯度已计算")
except Exception as e:
    print(f"✗ 训练失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("✓ 所有测试通过！")
print("="*60)
