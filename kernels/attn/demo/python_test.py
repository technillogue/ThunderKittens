import argparse
import sys
import time
import numpy as np
import torch
from typing import Optional

# Try to import the CUDA extension
try:
    import window_attn
except ImportError:
    print("Warning: Could not import window_attn CUDA module. Make sure it's built correctly.")
    window_attn = None

def create_sliding_window_mask(seq_len: int, window_size: int):
    """
    Create a sliding window attention mask.
    
    Args:
        seq_len: Sequence length
        window_size: Window size (should be odd for centered window)
    
    Returns:
        Tensor of shape [seq_len, seq_len] with True for positions within the window
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    half_window = window_size // 2
    
    for i in range(seq_len):
        window_start = max(0, i - window_size)
        window_end = min(seq_len, i + half_window + 1)
        mask[i, window_start:i+1] = True
    
    return mask

def reference_window_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    window_size: int, 
    scale: Optional[float] = None
) -> torch.Tensor:
    """
    Computes sliding window attention as a reference implementation.
    
    Args:
        q: Query tensor of shape [batch, heads, seq_len, head_dim]
        k: Key tensor of shape [batch, heads, seq_len, head_dim]
        v: Value tensor of shape [batch, heads, seq_len, head_dim]
        window_size: Window size for sliding window attention
        scale: Optional scaling factor (if None, uses 1/sqrt(head_dim))
    
    Returns:
        Output tensor of shape [batch, heads, seq_len, head_dim]
    """
    batch, heads, seq_len, head_dim = q.shape
    
    # Create sliding window mask
    attn_mask = create_sliding_window_mask(seq_len, window_size)
    attn_mask = attn_mask.to(q.device)
    
    # Apply mask by setting masked positions to -inf before softmax
    mask_value = torch.finfo(q.dtype).min
    
    # Use PyTorch's scaled dot product attention
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)
        
    # Manual implementation to exactly match CUDA kernel behavior
    q = q * scale  # Apply scale to query as in CUDA implementation
    attn = torch.matmul(q, k.transpose(-1, -2))  # [batch, heads, seq_len, seq_len]
    
    # Apply window mask
    attn_mask = ~attn_mask  # Invert mask: True means we mask it out
    attn.masked_fill_(attn_mask, float("-inf"))
    
    attn = torch.softmax(attn, dim=-1)
    output = torch.matmul(attn, v)  # [batch, heads, seq_len, head_dim]
    
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test for window attention correctness")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--heads", type=int, default=1, help="Number of attention heads")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--window", type=int, default=256, help="Window size")
    parser.add_argument("--runs", type=int, default=3, help="Number of timing runs")
    
    args = parser.parse_args()

    batch_size=args.batch 
    heads=args.heads 
    seq_len=args.seq_len 
    head_dim=args.head_dim 
    window_size=args.window
    num_runs=args.runs
    # """
    # Run correctness test comparing CUDA implementation with PyTorch reference.
    # Args:
    #     batch_size: Batch size
    #     heads: Number of attention heads
    #     seq_len: Sequence length
    #     head_dim: Head dimension
    #     window_size: Window size for sliding window attention
    #     num_runs: Number of runs for timing
    # """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Generate random input data
    q = torch.randn(batch_size, heads, seq_len, head_dim, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(batch_size, heads, seq_len, head_dim, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(batch_size, heads, seq_len, head_dim, dtype=torch.bfloat16, device='cuda')
    
    # Convert to bfloat16 for kernel input
    q_bf16 = q.to(torch.bfloat16)
    k_bf16 = k.to(torch.bfloat16)
    v_bf16 = v.to(torch.bfloat16)
    
    # Pre-allocate output tensor for CUDA kernel
    o_cuda = torch.zeros_like(q_bf16)
    
    print(f"Running test with batch_size={batch_size}, heads={heads}, seq_len={seq_len}, head_dim={head_dim}, window_size={window_size}")
    
    # Run the reference implementation
    print("Running PyTorch reference implementation...")
    torch.cuda.synchronize()
    start_time = time.time()
    o_ref = reference_window_attention(q, k, v, window_size)
    torch.cuda.synchronize()
    ref_time = time.time() - start_time
    print(f"Reference implementation took {ref_time:.4f} seconds")
    
    # Check if window_attn module is available
    if window_attn is None:
        print("Skipping CUDA implementation test since module is not available")
        sys.exit(1)
   
    # Run the CUDA implementation
    print("Running CUDA implementation...")
    
    # Warmup run
    window_attn.attn_fwd(o_cuda, q_bf16, k_bf16, v_bf16)
    
    # Timing runs
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        window_attn.attn_fwd(o_cuda, q_bf16, k_bf16, v_bf16)
    torch.cuda.synchronize()
    cuda_time = (time.time() - start_time) / num_runs
    print(f"CUDA implementation took {cuda_time:.4f} seconds per run (averaged over {num_runs} runs)")
    
    # Convert outputs to float32 for comparison
    o_cuda_fp32 = o_cuda.to(torch.float32)
    
    # Compute error metrics
    abs_diff = torch.abs(o_cuda_fp32 - o_ref)
    mean_abs_error = torch.mean(abs_diff).item()
    max_abs_error = torch.max(abs_diff).item()
    
    # Compute relative error excluding near-zero elements
    mask = torch.abs(o_ref) > 1e-5
    if mask.sum() > 0:
        rel_diff = abs_diff[mask] / torch.abs(o_ref[mask])
        mean_rel_error = torch.mean(rel_diff).item()
        max_rel_error = torch.max(rel_diff).item()
    else:
        mean_rel_error = float('nan')
        max_rel_error = float('nan')
    
    # Print results
    print("\nError metrics:")
    print(f"Mean absolute error: {mean_abs_error:.6e}")
    print(f"Max absolute error: {max_abs_error:.6e}")
    print(f"Mean relative error: {mean_rel_error:.6e}")
    print(f"Max relative error: {max_rel_error:.6e}")
    
    # Determine if test passes
    tolerance = 0.01  # Allow 1% relative error for bfloat16 precision
    if max_rel_error < tolerance:
        print("\nTest PASSED! ✅")
    else:
        print("\nTest FAILED! ❌")
        
    # Compute throughput metrics
    elements_per_run = batch_size * heads * seq_len * head_dim
    throughput_gbps = elements_per_run * 2 * 4 / (cuda_time * 1e9)  # 2 bytes per bfloat16, convert to GB/s
    
    # Calculate FLOPS
    # Each window attention involves:
    # - Query-Key matmul: 2 * batch * heads * seq_len * window_size * head_dim
    # - Softmax: 4 * batch * heads * seq_len * window_size
    # - Attention-Value matmul: 2 * batch * heads * seq_len * window_size * head_dim
    flops = (
        2 * batch_size * heads * seq_len * window_size * head_dim +  # QK^T
        4 * batch_size * heads * seq_len * window_size +            # Softmax
        2 * batch_size * heads * seq_len * window_size * head_dim    # (QK^T)V
    )
    tflops = flops / (cuda_time * 1e12)  # Convert to TFLOPs
    
    print(f"\nPerformance metrics:")
    print(f"Throughput: {throughput_gbps:.2f} GB/s")
    print(f"Efficiency: {tflops:.2f} TFLOPs")
    
    result = {
        "mean_abs_error": mean_abs_error,
        "max_abs_error": max_abs_error,
        "mean_rel_error": mean_rel_error,
        "max_rel_error": max_rel_error,
        "ref_time": ref_time,
        "cuda_time": cuda_time,
        "speedup": ref_time / cuda_time,
        "tflops": tflops
    }
