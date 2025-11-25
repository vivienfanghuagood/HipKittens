"""
Simple examples demonstrating HipKittens Attention Module usage.
These are minimal, runnable examples for common scenarios.
"""

import torch
from hipkittens_attn import HipKittensAttention, create_attention


def example_1_basic_inference():
    """Example 1: Basic inference with causal attention."""
    print("\n" + "="*60)
    print("Example 1: Basic Inference (Causal)")
    print("="*60)
    
    # Create attention module
    attn = HipKittensAttention(causal=True, training=False)
    
    # Input dimensions: Batch=8, SeqLen=2048, Heads=64, KV_Heads=8, Dim=128
    B, N, H, H_KV, D = 8, 2048, 64, 8, 128
    
    # Create random inputs
    q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda')
    
    # Run attention
    with torch.no_grad():
        output, lse = attn(q, k, v)
    
    print(f"Input:  Q={q.shape}, K={k.shape}, V={v.shape}")
    print(f"Output: {output.shape}")
    print(f"LSE:    {lse.shape}")
    print("✓ Success!")


def example_2_training_with_gradients():
    """Example 2: Training with backward pass."""
    print("\n" + "="*60)
    print("Example 2: Training with Gradients")
    print("="*60)
    
    # Create attention module for training
    attn = HipKittensAttention(causal=True, training=True)
    
    # Input dimensions
    B, N, H, H_KV, D = 4, 1024, 32, 8, 128
    
    # Create inputs with gradient tracking
    q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    k = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    v = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    
    # Forward pass
    output, lse = attn(q, k, v)
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    print(f"Output:  {output.shape}")
    print(f"Q grad:  {q.grad.shape}, norm={q.grad.norm().item():.4f}")
    print(f"K grad:  {k.grad.shape}, norm={k.grad.norm().item():.4f}")
    print(f"V grad:  {v.grad.shape}, norm={v.grad.norm().item():.4f}")
    print("✓ Success!")


def example_3_non_causal():
    """Example 3: Non-causal attention (BERT-style)."""
    print("\n" + "="*60)
    print("Example 3: Non-Causal Attention (BERT-style)")
    print("="*60)
    
    # Create non-causal attention
    attn = HipKittensAttention(causal=False, training=False)
    
    # BERT-base dimensions
    B, N, H, D = 16, 512, 12, 64
    
    # For standard MHA, set H_KV = H
    q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(B, N, H, D, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(B, N, H, D, dtype=torch.bfloat16, device='cuda')
    
    with torch.no_grad():
        output, _ = attn(q, k, v)
    
    print(f"BERT-base config: H={H}, D={D}")
    print(f"Output: {output.shape}")
    print("✓ Success!")


def example_4_factory_function():
    """Example 4: Using the factory function."""
    print("\n" + "="*60)
    print("Example 4: Factory Function")
    print("="*60)
    
    # Create different attention variants easily
    configs = [
        ("Causal Inference", {"causal": True, "training": False}),
        ("Non-Causal Training", {"causal": False, "training": True}),
    ]
    
    B, N, H, H_KV, D = 4, 512, 16, 4, 128
    
    for name, config in configs:
        print(f"\n{name}: {config}")
        attn = create_attention(**config)
        
        q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda')
        
        if config["training"]:
            q.requires_grad = True
            k.requires_grad = True
            v.requires_grad = True
        
        output, _ = attn(q, k, v)
        print(f"  Output: {output.shape} ✓")
    
    print("\n✓ All variants working!")


def example_5_gqa_variants():
    """Example 5: Different GQA configurations."""
    print("\n" + "="*60)
    print("Example 5: GQA Variants (MHA, GQA, MQA)")
    print("="*60)
    
    attn = HipKittensAttention(causal=True, training=False)
    
    # Test different GQA configurations
    configs = [
        ("Multi-Head Attention (MHA)", 32, 32),
        ("Grouped Query Attention (GQA)", 64, 8),
        ("Multi-Query Attention (MQA)", 32, 1),
    ]
    
    B, N, D = 4, 1024, 128
    
    for name, H, H_KV in configs:
        print(f"\n{name}: H={H}, H_KV={H_KV}")
        
        q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda')
        
        with torch.no_grad():
            output, _ = attn(q, k, v)
        
        print(f"  Group size: {H // H_KV}")
        print(f"  Output: {output.shape} ✓")
    
    print("\n✓ All GQA variants working!")


def example_6_llama_style():
    """Example 6: LLaMA-style model configuration."""
    print("\n" + "="*60)
    print("Example 6: LLaMA-Style Model")
    print("="*60)
    
    # LLaMA-3 8B configuration
    attn = HipKittensAttention(causal=True, training=True, return_lse=False)
    
    # Typical LLaMA dimensions
    B = 16        # Batch size
    N = 4096      # Context length
    H = 32        # Number of heads
    H_KV = 8      # GQA with 8 KV heads (4 groups)
    D = 128       # Head dimension
    
    print(f"LLaMA-3 config:")
    print(f"  Batch: {B}, Context: {N}")
    print(f"  Heads: {H}, KV Heads: {H_KV}, Dim: {D}")
    print(f"  Group size: {H // H_KV}")
    
    # Create inputs
    q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    k = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    v = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    
    # Forward pass
    output, _ = attn(q, k, v)
    
    # Backward pass
    loss = output.mean()
    loss.backward()
    
    print(f"\nForward:  {output.shape}")
    print(f"Backward: gradients computed ✓")
    print("✓ LLaMA-style attention working!")


def example_7_custom_nn_module():
    """Example 7: Integration with custom nn.Module."""
    print("\n" + "="*60)
    print("Example 7: Custom nn.Module Integration")
    print("="*60)
    
    import torch.nn as nn
    
    class SimpleAttentionLayer(nn.Module):
        def __init__(self, hidden_dim, num_heads, num_kv_heads):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = hidden_dim // num_heads
            
            # Linear projections
            self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
            self.out_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)
            
            # HipKittens attention
            self.attn = HipKittensAttention(causal=True, training=True, return_lse=False)
        
        def forward(self, x):
            B, N, _ = x.shape
            
            # Project
            q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim)
            v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim)
            
            # Attention
            attn_out, _ = self.attn(q, k, v)
            
            # Output projection
            attn_out = attn_out.reshape(B, N, -1)
            return self.out_proj(attn_out)
    
    # Create layer
    layer = SimpleAttentionLayer(
        hidden_dim=2048,
        num_heads=32,
        num_kv_heads=8
    ).cuda().bfloat16()
    
    # Test
    x = torch.randn(8, 512, 2048, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    output = layer(x)
    
    # Backward
    loss = output.sum()
    loss.backward()
    
    print(f"Input:  {x.shape}")
    print(f"Output: {output.shape}")
    print(f"Gradients computed: ✓")
    print("✓ Custom module integration working!")


def main():
    """Run all examples."""
    print("\n" + "█"*60)
    print("█  HipKittens Attention - Simple Examples")
    print("█"*60)
    
    if not torch.cuda.is_available():
        print("\n✗ ERROR: CUDA not available!")
        return
    
    print(f"\n✓ Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ PyTorch: {torch.__version__}\n")
    
    examples = [
        example_1_basic_inference,
        example_2_training_with_gradients,
        example_3_non_causal,
        example_4_factory_function,
        example_5_gqa_variants,
        example_6_llama_style,
        example_7_custom_nn_module,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n✗ Error in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
