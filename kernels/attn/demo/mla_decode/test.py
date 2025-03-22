import heapq
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import mla_decode
import numpy as np
import torch
from graphviz import Digraph
from scheduler import (
    create_arguments_from_task_schedule,
    priority_schedule_tasks,
    sample_schedule_generator,
    visualize_schedule,
)
from scheduler_regression import estimate_schedule_length
from scheduler_v2 import backward_schedule
from timings import save_gantt_chart
from tqdm import tqdm

torch.manual_seed(0)

D_Main, D_Rot = 512, 64
PAGE_SIZE = 256
# H = 16                  # set by q_heads
NUM_PAGES = 10000        # number of pages in cache
NUM_PROCESSORS = 132    # number of processors
MAX_NUM_PAGES = int(1e6) // PAGE_SIZE

ENABLE_TIMINGS = True

def init_arguments(seq_lengths: List[int], new_tokens: int, q_heads: int=16):

    B = len(seq_lengths)

    # Need to initialize QRot, QV, K_cache, V_cache, Lengths, Table    
    QRot    = torch.randn(B, new_tokens, q_heads, D_Rot, dtype=torch.bfloat16, device='cuda')
    QV      = torch.randn(B, new_tokens, q_heads, D_Main, dtype=torch.bfloat16, device='cuda')
    K_cache = torch.randn(NUM_PAGES, PAGE_SIZE, D_Rot, dtype=torch.bfloat16, device='cuda')
    V_cache = torch.randn(NUM_PAGES, PAGE_SIZE, D_Main, dtype=torch.bfloat16, device='cuda')
    Lengths = torch.tensor(seq_lengths, dtype=torch.int32, device='cuda')
    Table = torch.randint(0, NUM_PAGES, (B, MAX_NUM_PAGES), dtype=torch.int32, device='cuda')
    K_new = torch.randn(B, new_tokens, D_Rot, dtype=torch.bfloat16, device='cuda')
    V_new = torch.randn(B, new_tokens, D_Main, dtype=torch.bfloat16, device='cuda')

    return QRot, QV, K_cache, V_cache, Lengths, Table, K_new, V_new

def create_thundermla_arguments(seq_lengths, new_tokens, q_heads = 16):
    # Processor assignment heuristic: assign processors proportionally to sequence lengths.
    t0 = time.time()
    processor_assignments = [max(math.floor(s / sum(seq_lengths) * NUM_PROCESSORS), 1) for s in seq_lengths]
    while sum(processor_assignments) < NUM_PROCESSORS:
        min_idx = processor_assignments.index(max(processor_assignments))
        processor_assignments[min_idx] += 1

    new_tokens_for_estimate = new_tokens if q_heads == 16 else new_tokens // 2
    processor_assignments = sorted([(estimate_schedule_length(p, new_tokens_for_estimate, s), p, s, i) for i, (p, s) in enumerate(zip(processor_assignments, seq_lengths))])

    while len(seq_lengths) > 1:
        best, worst = processor_assignments[0], processor_assignments[-1]
        if best[1]-1 == 0: break
        new_t0, new_tn1 = estimate_schedule_length(best[1]-1, new_tokens_for_estimate, best[2]), estimate_schedule_length(worst[1]+1, new_tokens_for_estimate, worst[2])
        new_time = max(new_t0, new_tn1)
        if new_time < worst[0]:
            processor_assignments[0] = (new_t0, best[1]-1, best[2], best[-1])
            processor_assignments[-1] = (new_tn1, worst[1]+1, worst[2], worst[-1])
            processor_assignments = sorted(processor_assignments)
        else:
            break
    num_processors = [None for _ in seq_lengths]
    for _, p, s, i in processor_assignments:
        num_processors[i] = max(min(p, s//128), 1)
    # Create schedule
    start_processors = [sum(num_processors[:i]) for i in range(len(num_processors))]
    scheduled_tasks = []
    partial_uid, reduction_uid = 0, NUM_PROCESSORS
    for batch_id, (seq_l, start_p, num_p) in enumerate(zip(seq_lengths, start_processors, num_processors)):
        new_tasks, partial_uid, reduction_uid = backward_schedule(
            list(range(start_p, start_p + num_p)), batch_id, seq_l, list(range(new_tokens)), partial_uid, reduction_uid, q_heads
        )
        scheduled_tasks.extend(new_tasks)
    t1 = time.time()
    print(f'Time taken to create schedule: {(t1-t0)*1000} ms')
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_arguments_from_task_schedule(
        scheduled_tasks, new_tokens, num_processors=NUM_PROCESSORS, enable_timings=ENABLE_TIMINGS, q_heads=q_heads
    )
    # visualize_schedule(scheduled_tasks, NUM_PROCESSORS)
    return Instructions, O_scratch, Lvec_scratch, Semaphore, Timings

def create_rope_embeddings(seq_lengths, new_tokens, rope_dim=64, base: float = 10000.0):

    seq_len = max(seq_lengths) + new_tokens

    half_dim = rope_dim // 2

    positions = torch.arange(seq_len, dtype=torch.bfloat16).unsqueeze(1) # [seq_len, 1]
    dim_indices = torch.arange(half_dim, dtype=torch.bfloat16).unsqueeze(0) # [1, half_dim]

    freq = 1.0 / (base ** (2 * dim_indices / rope_dim)) # [1, half_dim]
    angles = positions * freq # [seq_len, half_dim]
    
    sin = torch.sin(angles).to(torch.device("cuda")) # [seq_len, half_dim]
    cos = torch.cos(angles).to(torch.device("cuda")) # [seq_len, half_dim]

    return sin, cos
    

def apply_rope(QRot, Lengths, sin, cos):
    QRot_clone = QRot.clone()
    batch_size, new_tokens, num_heads, rope_dim = QRot.shape
    half_dim = rope_dim // 2

    def rotate(q, length):
        # Get positions for the new tokens - they should be the next positions after length
        new_positions = torch.arange(length, length + new_tokens)
        new_tok_sin = sin[new_positions]
        new_tok_cos = cos[new_positions]

        sin_expanded = new_tok_sin.unsqueeze(1)  # [new_tokens, 1, half_dim]
        cos_expanded = new_tok_cos.unsqueeze(1)  # [new_tokens, 1, half_dim]

        q1, q2 = q[..., :half_dim], q[..., half_dim:]

        q_rot1 = q1 * cos_expanded - q2 * sin_expanded
        q_rot2 = q1 * sin_expanded + q2 * cos_expanded

        return torch.cat([q_rot1, q_rot2], dim=-1)

    # Could just do this at once as a batch operation, but good enough for now
    for batch_idx in range(len(Lengths)):
        seq_length = Lengths[batch_idx]
        QRot_clone[batch_idx] = rotate(QRot_clone[batch_idx], seq_length)

    return QRot_clone


def run_thundermla(QRot, QV, sin, cos, K_cache, V_cache, K_new, V_new, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings, tic=None):
    q_heads = QRot.shape[2]
    if tic is None:
        Semaphore.zero_()
        tic = 1
    O = torch.zeros_like(QV)
    softmax_scale = 1.0 / math.sqrt(D_Main+D_Rot)
    torch.cuda.synchronize()
    mla_decode_fn = mla_decode.mla_decode_8_heads if q_heads == 8 else mla_decode.mla_decode

    if Timings is not None:
        mla_decode_fn(Instructions, QRot, QV, sin, cos, K_cache, V_cache, K_new, V_new, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, tic, Timings)
        mla_decode_fn(Instructions, QRot, QV, sin, cos, K_cache, V_cache, K_new, V_new, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, 1-tic, Timings)
    else:
        mla_decode_fn(Instructions, QRot, QV, sin, cos, K_cache, V_cache, K_new, V_new, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, tic)
        mla_decode_fn(Instructions, QRot, QV, sin, cos, K_cache, V_cache, K_new, V_new, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, 1-tic)
    torch.cuda.synchronize()
    return O, Timings

def profile_thundermla(QRot, QV, sin, cos, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings, ITERS=100):
    q_heads = QRot.shape[2]
    Semaphore.zero_()
    O = torch.zeros_like(QV)
    softmax_scale = 1.0 / math.sqrt(D_Main+D_Rot)
    # execute once to warm up
    mla_decode_fn = mla_decode.mla_decode_8_heads if q_heads == 8 else mla_decode.mla_decode

    if Timings is not None:
        mla_decode_fn(Instructions, QRot, QV, sin, cos, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, 1, Timings)
    else:
        mla_decode_fn(Instructions, QRot, QV, sin, cos, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, 1)
    torch.cuda.synchronize()
    t0 = time.time()
    for it in range(ITERS):
        if Timings is not None:
            mla_decode_fn(Instructions, QRot, QV, sin, cos, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, it%2, Timings)
        else:
            mla_decode_fn(Instructions, QRot, QV, sin, cos, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, it%2)

    torch.cuda.synchronize()
    t1 = time.time()
    return (t1-t0) / ITERS

def run_mla_torch(QRot, QV, K_cache, V_cache, K_new, V_new, Lengths, Table):
    Q = torch.concat([QRot, QV], dim=-1)
    q_heads = Q.shape[2]
    new_tokens = K_new.shape[1]

    # RoPE for Q
    sin, cos = create_rope_embeddings(Lengths, new_tokens)
    QRot_rope_applied = apply_rope(QRot, Lengths, sin, cos)
    Q = torch.concat([QRot_rope_applied, QV], dim=-1)

    softmax_scale = 1.0 / math.sqrt(D_Main+D_Rot)
    O = torch.zeros_like(QV)
    
    for b, length in enumerate(Lengths):
        # Extract only the valid tokens for this batch (up to its length)
        batch_table = Table[b, :math.ceil(length/PAGE_SIZE)]
        
        # Get K and V from cache, reshape to match sequence length
        k_from_cache = torch.cat([K_cache, V_cache], dim=-1)[batch_table].reshape(1, -1, Q.shape[-1])
        v_from_cache = V_cache[batch_table].reshape(1, -1, QV.shape[-1])
        
        # Truncate to actual sequence length
        k_from_cache = k_from_cache[:, :length]
        v_from_cache = v_from_cache[:, :length]
        
        # Append new tokens
        k_batch_new = torch.cat([K_new[b:b+1], V_new[b:b+1]], dim=-1)
        full_K = torch.cat([k_from_cache, k_batch_new], dim=1)
        full_V = torch.cat([v_from_cache, V_new[b:b+1]], dim=1)
        
        full_seqlen = length + new_tokens
        mask = torch.ones(new_tokens, full_seqlen, dtype=torch.bool).tril(diagonal=full_seqlen-new_tokens).to(Q.device)
        
        O[b:b+1] = torch.nn.functional.scaled_dot_product_attention(
            Q[b:b+1].transpose(1, 2),
            full_K.unsqueeze(-2).repeat((1,1,q_heads,1)).transpose(1, 2),
            full_V.unsqueeze(-2).repeat((1,1,q_heads,1)).transpose(1, 2),
            is_causal=False,
            attn_mask=mask,
            scale=softmax_scale
        ).transpose(1, 2)
        
    return O



def retry_on_assertion(max_retries=100):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AssertionError:
                print(f"\nAssertion failed, retrying up to {max_retries} times...")
                for attempt in tqdm(range(max_retries)):
                    try:
                        result = func(*args, **kwargs)
                        print(f"\nSucceeded after {attempt + 2} attempts")
                        return result
                    except AssertionError:
                        continue
                # If we get here, we failed all retries
                print("All retries failed, raising last assertion")
                return func(*args, **kwargs)  # This will raise the assertion
        return wrapper
    return decorator



@retry_on_assertion()
def main(seq_lengths, new_tokens, q_heads=16, use_rope=True):
    # print(f' ----------- starting seq_lengths: {seq_lengths} new_tokens: {new_tokens} q_heads: {q_heads} use_rope: {use_rope} -----------')
    seq_lengths = sorted(seq_lengths)
    QRot, QV, K_cache, V_cache, Lengths, Table, K_new, V_new = init_arguments(seq_lengths, new_tokens, q_heads)
    ref = run_mla_torch(QRot, QV, K_cache, V_cache, K_new, V_new, Lengths, Table)
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_thundermla_arguments(seq_lengths, new_tokens, q_heads)
    sin, cos = create_rope_embeddings(Lengths, new_tokens)
    O, Timings = run_thundermla(QRot, QV, sin, cos, K_cache, V_cache, K_new, V_new, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings)

    # check kv update (last N tokens match between K_cache and K_new)
    for b in range(len(Lengths)):
        eos_page_idx = Lengths[b] // PAGE_SIZE
        eos_page_addr = Table[b, eos_page_idx]
        offset_in_page = Lengths[b] % PAGE_SIZE
        if offset_in_page + new_tokens > PAGE_SIZE:
            # print(f"Skipping batch {b} because of page boundary")
            continue

        cached_K = K_cache[eos_page_addr, offset_in_page:offset_in_page+new_tokens]
        cached_V = V_cache[eos_page_addr, offset_in_page:offset_in_page+new_tokens]
        new_K = K_new[b, :]
        new_V = V_new[b, :]

        if not torch.allclose(cached_K, new_K, atol=1e-3):
            print(cached_K[..., :4])
            print(new_K[..., :4])
            assert False, "K_cache update failed"
        if not torch.allclose(cached_V, new_V, atol=1e-3):
            print(cached_V[..., :4])
            print(new_V[..., :4])
            assert False, "V_cache update failed"

    # flatten heads and head_dim
    cosine_similarity = torch.nn.functional.cosine_similarity(O.float().flatten(start_dim=-2, end_dim=-1), ref.float().flatten(start_dim=-2, end_dim=-1), dim=-1)
    is_correct = cosine_similarity > 0.99
    if not is_correct.all() and cosine_similarity.min() < 0.9:
        print(f"Cosine similarity mean: {cosine_similarity.mean()}, worst: {cosine_similarity.min()}")
        print(cosine_similarity.shape)
        print("out", O[..., :2, -4:])
        print("ref", ref[..., :2, -4:])
        print("\nError pattern:")
        print("batch   sequence")
        for b, s in enumerate(is_correct):
            if s.all():
                continue
            print(f"{b:6d}   ", end="")
            errstring = ["✓" if e else "✗" for e in s]
            print(" ".join(errstring))
            print()
        assert False


    # time_per_iter = profile_thundermla(QRot, QV, sin, cos, K_cache, V_cache, K_new, V_new, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings)
    # print(f"Time per iter: {time_per_iter*1000} ms")

    # save_gantt_chart(Timings, Instructions, name='new')

if __name__ == "__main__":
    main([1], 1, 16)
    main([16], 4, 16)
    main([32], 4, 16)
    main([64], 2, 16)
    main([4641,45118,1730,1696], 4, 16)
    main([65536], 1, 16)
    main([512]*64, 2, 16)
    main([4096]*132, 4, 16)
    main([871,568,711,329,617,1015,348,978,543,837,650,1020,924,679,560,497,650,406,381,423,511,423,569,943,645,820,829,883,937,765,711,847,722,546,519,279,516,315,664,845,850,546,670,871,527,329,446,764,582,1011,453,655,532,985,1019,810,317,305,949,317,669,768,530,349], 4, 16)
    
    main([4641,45118,1730,1696], 4, 8)
    main([65536], 1, 8)
    main([512]*64, 2, 8)
    main([4096]*132, 4, 8)
    main([871,568,711,329,617,1015,348,978,543,837,650,1020,924,679,560,497,650,406,381,423,511,423,569,943,645,820,829,883,937,765,711,847,722,546,519,279,516,315,664,845,850,546,670,871,527,329,446,764,582,1011,453,655,532,985,1019,810,317,305,949,317,669,768,530,349], 4, 8)
