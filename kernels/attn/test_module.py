"""
Test script for HipKittens Attention Module
This script demonstrates various usage scenarios of the attention module.
Note: This script requires compiled kernels to run.
"""

import torch
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from hipkittens_attn import (
    HipKittensAttention,
    HipKittensAttentionJIT,
    create_attention
)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")


def test_basic_forward():
    """Test basic forward pass with non-causal attention."""
    print_section("Test 1: Basic Forward Pass (Non-Causal)")
    
    try:
        attn = HipKittensAttention(
            causal=False,
            training=False,
            return_lse=True
        )
        
        B, N, H, H_KV, D = 8, 1024, 32, 8, 128
        print(f"Dimensions: B={B}, N={N}, H={H}, H_KV={H_KV}, D={D}")
        
        q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda')
        
        with torch.no_grad():
            output, lse = attn(q, k, v)
        
        print(f"✓ Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ LSE shape: {lse.shape}")
        print(f"✓ Output dtype: {output.dtype}")
        print(f"✓ LSE dtype: {lse.dtype}")
        print("✓ Test PASSED")
        
    except Exception as e:
        print(f"✗ Test FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_causal_forward():
    """Test forward pass with causal masking."""
    print_section("Test 2: Causal Forward Pass")
    
    try:
        attn = HipKittensAttention(
            causal=True,
            training=False,
            return_lse=True
        )
        
        B, N, H, H_KV, D = 4, 2048, 64, 8, 128
        print(f"Dimensions: B={B}, N={N}, H={H}, H_KV={H_KV}, D={D}")
        
        q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda')
        
        with torch.no_grad():
            output, lse = attn(q, k, v)
        
        print(f"✓ Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ LSE shape: {lse.shape}")
        print("✓ Test PASSED")
        
    except Exception as e:
        print(f"✗ Test FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_backward_pass():
    """Test backward pass with gradient computation."""
    print_section("Test 3: Backward Pass with Gradients")
    
    try:
        attn = HipKittensAttention(
            causal=True,
            training=True,
            return_lse=True
        )
        
        B, N, H, H_KV, D = 8, 1024, 32, 8, 128
        print(f"Dimensions: B={B}, N={N}, H={H}, H_KV={H_KV}, D={D}")
        
        q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device='cuda', requires_grad=True)
        k = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda', requires_grad=True)
        v = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda', requires_grad=True)
        
        # Forward
        output, lse = attn(q, k, v)
        
        # Backward
        loss = output.sum()
        loss.backward()
        
        print(f"✓ Forward output shape: {output.shape}")
        print(f"✓ Q gradient shape: {q.grad.shape}")
        print(f"✓ K gradient shape: {k.grad.shape}")
        print(f"✓ V gradient shape: {v.grad.shape}")
        print(f"✓ Q grad norm: {q.grad.norm().item():.4f}")
        print(f"✓ K grad norm: {k.grad.norm().item():.4f}")
        print(f"✓ V grad norm: {v.grad.norm().item():.4f}")
        print("✓ Test PASSED")
        
    except Exception as e:
        print(f"✗ Test FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_different_head_dims():
    """Test with different head dimensions."""
    print_section("Test 4: Different Head Dimensions")
    
    head_dims = [64, 128]
    
    for D in head_dims:
        try:
            print(f"\nTesting D={D}:")
            attn = HipKittensAttention(causal=False, training=False)
            
            B, N, H, H_KV = 4, 512, 32, 8
            q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device='cuda')
            k = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda')
            v = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda')
            
            with torch.no_grad():
                output, lse = attn(q, k, v)
            
            print(f"  ✓ Output shape: {output.shape}")
            print(f"  ✓ Head dim {D} working correctly")
            
        except Exception as e:
            print(f"  ✗ Failed for D={D}: {e}")


def test_factory_function():
    """Test the create_attention factory function."""
    print_section("Test 5: Factory Function")
    
    try:
        # Test different configurations
        configs = [
            {"causal": False, "training": False},
            {"causal": True, "training": False},
            {"causal": False, "training": True},
            {"causal": True, "training": True},
        ]
        
        for i, config in enumerate(configs):
            print(f"\nConfig {i+1}: {config}")
            attn = create_attention(**config)
            
            B, N, H, H_KV, D = 4, 512, 16, 4, 128
            q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device='cuda')
            k = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda')
            v = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda')
            
            if config["training"]:
                q.requires_grad = True
                k.requires_grad = True
                v.requires_grad = True
            
            output, lse = attn(q, k, v)
            print(f"  ✓ Output shape: {output.shape}")
        
        print("\n✓ All factory configurations PASSED")
        
    except Exception as e:
        print(f"✗ Test FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_input_validation():
    """Test input validation and error handling."""
    print_section("Test 6: Input Validation")
    
    attn = HipKittensAttention(causal=False, training=False)
    
    test_cases = [
        {
            "name": "Wrong Q shape (3D instead of 4D)",
            "q_shape": (8, 1024, 128),
            "k_shape": (8, 1024, 8, 128),
            "v_shape": (8, 1024, 8, 128),
        },
        {
            "name": "Mismatched batch sizes",
            "q_shape": (8, 1024, 32, 128),
            "k_shape": (16, 1024, 8, 128),
            "v_shape": (16, 1024, 8, 128),
        },
        {
            "name": "Mismatched sequence lengths",
            "q_shape": (8, 1024, 32, 128),
            "k_shape": (8, 2048, 8, 128),
            "v_shape": (8, 2048, 8, 128),
        },
        {
            "name": "Invalid GQA (H not divisible by H_KV)",
            "q_shape": (8, 1024, 31, 128),
            "k_shape": (8, 1024, 8, 128),
            "v_shape": (8, 1024, 8, 128),
        },
    ]
    
    for test_case in test_cases:
        try:
            print(f"\nTesting: {test_case['name']}")
            q = torch.randn(*test_case['q_shape'], dtype=torch.bfloat16, device='cuda')
            k = torch.randn(*test_case['k_shape'], dtype=torch.bfloat16, device='cuda')
            v = torch.randn(*test_case['v_shape'], dtype=torch.bfloat16, device='cuda')
            
            output, _ = attn(q, k, v)
            print(f"  ✗ Should have raised an error!")
            
        except (AssertionError, RuntimeError) as e:
            print(f"  ✓ Correctly caught error: {str(e)[:60]}...")


def test_gqa_variants():
    """Test different GQA configurations (MHA, GQA, MQA)."""
    print_section("Test 7: GQA Variants")
    
    attn = HipKittensAttention(causal=False, training=False)
    
    configurations = [
        {"name": "Multi-Head Attention (MHA)", "H": 32, "H_KV": 32},
        {"name": "Grouped Query Attention (GQA)", "H": 64, "H_KV": 8},
        {"name": "Multi-Query Attention (MQA)", "H": 32, "H_KV": 1},
    ]
    
    B, N, D = 4, 1024, 128
    
    for config in configurations:
        try:
            print(f"\n{config['name']} (H={config['H']}, H_KV={config['H_KV']}):")
            
            q = torch.randn(B, N, config['H'], D, dtype=torch.bfloat16, device='cuda')
            k = torch.randn(B, N, config['H_KV'], D, dtype=torch.bfloat16, device='cuda')
            v = torch.randn(B, N, config['H_KV'], D, dtype=torch.bfloat16, device='cuda')
            
            with torch.no_grad():
                output, _ = attn(q, k, v)
            
            print(f"  ✓ Output shape: {output.shape}")
            print(f"  ✓ Configuration working correctly")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")


def test_jit_compilation():
    """Test JIT compilation support."""
    print_section("Test 8: JIT Compilation")
    
    try:
        print("Creating JIT-compilable module...")
        attn = HipKittensAttentionJIT(
            causal=True,
            training=False,
            max_seq_len=4096,
            head_dims=[64, 128]
        )
        
        print("Attempting to JIT script...")
        attn_jit = torch.jit.script(attn)
        print("✓ JIT scripting successful")
        
        # Test with D=128
        B, N, H, H_KV, D = 4, 1024, 32, 8, 128
        q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda')
        
        output, lse = attn_jit(q, k, v)
        print(f"✓ JIT module forward pass successful")
        print(f"✓ Output shape: {output.shape}")
        
        # Save and load
        print("Testing save/load...")
        torch.jit.save(attn_jit, "/tmp/hipkittens_attn_jit.pt")
        loaded_attn = torch.jit.load("/tmp/hipkittens_attn_jit.pt")
        output2, _ = loaded_attn(q, k, v)
        print("✓ Save/load successful")
        
        print("✓ JIT Test PASSED")
        
    except Exception as e:
        print(f"✗ JIT Test FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_performance_benchmark():
    """Simple performance benchmark."""
    print_section("Test 9: Performance Benchmark")
    
    try:
        import time
        
        attn = HipKittensAttention(causal=True, training=False)
        
        B, N, H, H_KV, D = 16, 2048, 64, 8, 128
        print(f"Benchmark config: B={B}, N={N}, H={H}, H_KV={H_KV}, D={D}")
        
        q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device='cuda')
        
        # Warmup
        print("Warming up...")
        for _ in range(100):
            with torch.no_grad():
                _ = attn(q, k, v)
        
        # Benchmark
        num_iters = 1000
        print(f"Running {num_iters} iterations...")
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(num_iters):
            with torch.no_grad():
                _ = attn(q, k, v)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        avg_time_ms = (end - start) / num_iters * 1000
        
        print(f"\n✓ Average time: {avg_time_ms:.4f} ms")
        
        # Rough FLOPS estimate (causal attention)
        flops = 4 * B * N * N * H * D // 2  # Divide by 2 for causal
        tflops = (flops / 1e12) / (avg_time_ms / 1000)
        print(f"✓ Estimated performance: {tflops:.2f} TFLOPS")
        
    except Exception as e:
        print(f"✗ Benchmark FAILED: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█" + "  HipKittens Attention Module - Test Suite".center(58) + "█")
    print("█" + " "*58 + "█")
    print("█"*60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n✗ ERROR: CUDA is not available!")
        print("  This module requires a CUDA-enabled GPU.")
        return
    
    print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✓ PyTorch version: {torch.__version__}")
    
    # Run tests
    tests = [
        test_basic_forward,
        test_causal_forward,
        test_backward_pass,
        test_different_head_dims,
        test_factory_function,
        test_input_validation,
        test_gqa_variants,
        test_jit_compilation,
        test_performance_benchmark,
    ]
    
    for test_func in tests:
        try:
            test_func()
        except KeyboardInterrupt:
            print("\n\nTests interrupted by user.")
            break
        except Exception as e:
            print(f"\n✗ Unexpected error in {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print_section("Test Suite Complete")
    print("All tests finished. Check output above for results.\n")


if __name__ == "__main__":
    main()
