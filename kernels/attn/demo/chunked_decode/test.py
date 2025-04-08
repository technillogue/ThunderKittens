import math
import time
from typing import List

import gqa_decode
import numpy as np
import torch
from flash_attn.layers.rotary import apply_rotary_emb_torch
from scheduler import create_arguments_from_task_schedule, visualize_schedule
from scheduler_regression import estimate_schedule_length
from scheduler_v2 import backward_schedule
from timings import save_gantt_chart
from tqdm import tqdm

D = 128
PAGE_SIZE = 256
NUM_PAGES = 10000  # number of pages in cache
NUM_PROCESSORS = 132  # number of processors
seed = torch.randint(1000000, (1,)).item()

ENABLE_TIMINGS = True


def init_arguments(seq_lengths: List[int], new_tokens: int, q_heads: int = 8):
    # fix the seed on every iteration so that the each test gets the same initial conditions
    torch.manual_seed(seed)

    B = len(seq_lengths)
    max_pages_in_batch = (
        math.ceil((max(seq_lengths) / PAGE_SIZE)) + 10
    )  # hold new tokens in cache
    assert B * max_pages_in_batch <= NUM_PAGES, (
        f"B * max_pages_in_batch = {B * max_pages_in_batch} > NUM_PAGES = {NUM_PAGES}"
    )

    # Need to initialize Q, K_cache, V_cache, Lengths, Table
    Q = torch.randn(B, new_tokens, q_heads, D, dtype=torch.bfloat16, device="cuda")
    K_cache = torch.randn(NUM_PAGES, PAGE_SIZE, D, dtype=torch.bfloat16, device="cuda")
    V_cache = torch.randn(NUM_PAGES, PAGE_SIZE, D, dtype=torch.bfloat16, device="cuda")
    Lengths = torch.tensor(seq_lengths, dtype=torch.int32, device="cuda")
    Table = torch.randperm(NUM_PAGES, device="cuda", dtype=torch.int32)[
        : B * max_pages_in_batch
    ].reshape(B, max_pages_in_batch)
    K_new = torch.randn(B, new_tokens, D, dtype=torch.bfloat16, device="cuda")
    V_new = torch.randn(B, new_tokens, D, dtype=torch.bfloat16, device="cuda")

    return Q, K_cache, V_cache, Lengths, Table, K_new, V_new


def create_thundergqa_arguments(seq_lengths, new_tokens, q_heads=8):
    # Processor assignment heuristic: assign processors proportionally to sequence lengths.
    t0 = time.time()
    processor_assignments = [
        max(math.floor(s / sum(seq_lengths) * NUM_PROCESSORS), 1) for s in seq_lengths
    ]
    while sum(processor_assignments) < NUM_PROCESSORS:
        min_idx = processor_assignments.index(max(processor_assignments))
        processor_assignments[min_idx] += 1

    new_tokens_for_estimate = new_tokens // 2  # TODO: check
    processor_assignments = sorted(
        [
            (estimate_schedule_length(p, new_tokens_for_estimate, s), p, s, i)
            for i, (p, s) in enumerate(zip(processor_assignments, seq_lengths))
        ]
    )

    while len(seq_lengths) > 1:
        best, worst = processor_assignments[0], processor_assignments[-1]
        if best[1] - 1 == 0:
            break
        new_t0, new_tn1 = (
            estimate_schedule_length(best[1] - 1, new_tokens_for_estimate, best[2]),
            estimate_schedule_length(worst[1] + 1, new_tokens_for_estimate, worst[2]),
        )
        new_time = max(new_t0, new_tn1)
        if new_time < worst[0]:
            processor_assignments[0] = (new_t0, best[1] - 1, best[2], best[-1])
            processor_assignments[-1] = (new_tn1, worst[1] + 1, worst[2], worst[-1])
            processor_assignments = sorted(processor_assignments)
        else:
            break
    num_processors = [None for _ in seq_lengths]
    for _, p, s, i in processor_assignments:
        num_processors[i] = max(min(p, s // 128), 1)
    # Create schedule
    start_processors = [sum(num_processors[:i]) for i in range(len(num_processors))]
    scheduled_tasks = []
    partial_uid, reduction_uid = 0, NUM_PROCESSORS
    for batch_id, (seq_l, start_p, num_p) in enumerate(
        zip(seq_lengths, start_processors, num_processors)
    ):
        new_tasks, partial_uid, reduction_uid = backward_schedule(
            list(range(start_p, start_p + num_p)),
            batch_id,
            seq_l,
            list(range(new_tokens)),
            partial_uid,
            reduction_uid,
            q_heads,
        )
        scheduled_tasks.extend(new_tasks)
    t1 = time.time()
    print(f"Time taken to create schedule: {(t1 - t0) * 1000} ms")
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = (
        create_arguments_from_task_schedule(
            scheduled_tasks,
            new_tokens,
            num_processors=NUM_PROCESSORS,
            enable_timings=ENABLE_TIMINGS,
            q_heads=q_heads,
        )
    )
    # visualize_schedule(scheduled_tasks, NUM_PROCESSORS)
    return Instructions, O_scratch, Lvec_scratch, Semaphore, Timings


# https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/layers/rotary.py
def create_rope_embeddings(seq_lengths, new_tokens, rope_dim, base: float = 10000.0):
    # add NUM_ROWS_d2 to account for loading in 16 rows to not overflow
    NUM_ROWS_d2 = 16
    seq_len = max(seq_lengths) + new_tokens + NUM_ROWS_d2

    t = torch.arange(seq_len, device=torch.device("cuda"), dtype=torch.float32)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(
                0, rope_dim, 2, device=torch.device("cuda"), dtype=torch.float32
            )
            / rope_dim
        )
    )

    freqs = torch.outer(t, inv_freq)

    cos = torch.cos(freqs).to(torch.bfloat16)
    sin = torch.sin(freqs).to(torch.bfloat16)

    return cos, sin


def apply_rope(X, Lengths, cos, sin):
    assert X.ndim == 3 or X.ndim == 4

    X_rope = X.clone()
    is_k = X_rope.ndim == 3

    if is_k:
        X_rope = X_rope.unsqueeze(2)

    _, new_tokens, _, rope_dim = X_rope.shape

    for i in range(len(Lengths)):
        X_rope[i] = apply_rotary_emb_torch(
            X_rope[i],
            cos[..., Lengths[i] : Lengths[i] + new_tokens, :],
            sin[..., Lengths[i] : Lengths[i] + new_tokens, :],
            interleaved=False,
        )

    return X_rope.squeeze(2) if is_k else X_rope


def run_thundergqa(
    Q,
    K_cache,
    V_cache,
    K_new,
    V_new,
    sin,
    cos,
    Table,
    Instructions,
    O_scratch,
    Lvec_scratch,
    Semaphore,
    Timings,
    tic=None,
):
    q_heads = Q.shape[2]
    if tic is None:
        Semaphore.zero_()
        tic = 1
    O = torch.zeros_like(Q)
    softmax_scale = 1.0 / math.sqrt(D)
    torch.cuda.synchronize()
    assert q_heads == 8
    gqa_decode_fn = gqa_decode.gqa_decode_8_heads
    gqa_decode_fn(
        Instructions,
        Q,
        K_cache,
        V_cache,
        K_new,
        V_new,
        sin,
        cos,
        Table,
        O,
        O_scratch,
        Lvec_scratch,
        Semaphore,
        softmax_scale,
        tic,
        Timings,
    )
    gqa_decode_fn(
        Instructions,
        Q,
        K_cache,
        V_cache,
        K_new,
        V_new,
        sin,
        cos,
        Table,
        O,
        O_scratch,
        Lvec_scratch,
        Semaphore,
        softmax_scale,
        1 - tic,
        Timings,
    )
    torch.cuda.synchronize()
    return O, Timings


def profile_thundergqa(
    Q,
    K_cache,
    V_cache,
    K_new,
    V_new,
    sin,
    cos,
    Table,
    Instructions,
    O_scratch,
    Lvec_scratch,
    Semaphore,
    Timings,
    ITERS=100,
):
    q_heads = Q.shape[2]
    Semaphore.zero_()
    O = torch.zeros_like(Q)
    softmax_scale = 1.0 / math.sqrt(D)
    # execute once to warm up
    assert q_heads == 8
    gqa_decode_fn = gqa_decode.gqa_decode_8_heads
    gqa_decode_fn(
        Instructions,
        Q,
        K_cache,
        V_cache,
        K_new,
        V_new,
        sin,
        cos,
        Table,
        O,
        O_scratch,
        Lvec_scratch,
        Semaphore,
        softmax_scale,
        1,
        Timings,
    )
    torch.cuda.synchronize()
    t0 = time.time()
    for it in range(ITERS):
        gqa_decode_fn(
            Instructions,
            Q,
            K_cache,
            V_cache,
            K_new,
            V_new,
            sin,
            cos,
            Table,
            O,
            O_scratch,
            Lvec_scratch,
            Semaphore,
            softmax_scale,
            it % 2,
            Timings,
        )

    torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / ITERS


def run_gqa_torch(Q, K_cache, V_cache, K_new, V_new, cos, sin, Lengths, Table):
    q_heads = Q.shape[2]
    new_tokens = K_new.shape[1]

    # RoPE for Q
    Q = apply_rope(Q, Lengths, cos, sin)

    # RoPE for K
    K_new_rope_applied = apply_rope(K_new, Lengths, cos, sin)

    softmax_scale = 1.0 / math.sqrt(D)
    O = torch.zeros_like(Q)

    for b, length in enumerate(Lengths):
        # Extract only the valid tokens for this batch (up to its length)
        batch_table = Table[b, : math.ceil(length / PAGE_SIZE)]

        # Get K and V from cache, reshape to match sequence length
        k_from_cache = K_cache[batch_table].reshape(1, -1, Q.shape[-1])
        v_from_cache = V_cache[batch_table].reshape(1, -1, Q.shape[-1])

        # b, -1, d

        # Truncate to actual sequence length
        k_from_cache = k_from_cache[:, :length]
        v_from_cache = v_from_cache[:, :length]

        # Append new tokens
        full_K = torch.cat([k_from_cache, K_new_rope_applied[b : b + 1]], dim=1)
        full_V = torch.cat([v_from_cache, V_new[b : b + 1]], dim=1)

        full_seqlen = length + new_tokens
        mask = (
            torch.ones(new_tokens, full_seqlen, dtype=torch.bool)
            .tril(diagonal=full_seqlen - new_tokens)
            .to(Q.device)
        )

        O[b : b + 1] = torch.nn.functional.scaled_dot_product_attention(
            Q[b : b + 1].transpose(1, 2),
            full_K.unsqueeze(-2).repeat((1, 1, q_heads, 1)).transpose(1, 2),
            full_V.unsqueeze(-2).repeat((1, 1, q_heads, 1)).transpose(1, 2),
            is_causal=False,
            attn_mask=mask,
            scale=softmax_scale,
        ).transpose(1, 2)

    return O, K_new_rope_applied


def retry_on_assertion(max_retries=5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AssertionError:
                print(f"\nAssertion failed, retrying up to {max_retries} times...")
                for attempt in tqdm(range(max_retries)):
                    try:
                        result = func(*args, **kwargs)
                        print(f"\nSucceeded after {attempt + 1} failures")
                        return result
                    except AssertionError:
                        continue
                # If we get here, we failed all retries
                print("All retries failed, raising last assertion")
                return func(*args, **kwargs)  # This will raise the assertion

        return wrapper

    return decorator


def print_rounded(tensor, shape, decimals=4):
    return torch.round(tensor.view(shape), decimals=decimals)


@retry_on_assertion()
def main(seq_lengths, new_tokens, q_heads=8):
    print(
        f" ----------- batches: {len(seq_lengths)} mean seq_length: {np.mean(seq_lengths)} new_tokens: {new_tokens} q_heads: {q_heads} -----------"
    )
    seq_lengths = sorted(seq_lengths)
    Q, K_cache, V_cache, Lengths, Table, K_new, V_new = init_arguments(
        seq_lengths, new_tokens, q_heads
    )

    cos, sin = create_rope_embeddings(Lengths, new_tokens, rope_dim=D)

    ref, K_new_rope_applied = run_gqa_torch(
        Q, K_cache, V_cache, K_new, V_new, cos, sin, Lengths, Table
    )
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = (
        create_thundergqa_arguments(seq_lengths, new_tokens, q_heads)
    )
    O, Timings = run_thundergqa(
        Q,
        K_cache,
        V_cache,
        K_new,
        V_new,
        cos,
        sin,
        Table,
        Instructions,
        O_scratch,
        Lvec_scratch,
        Semaphore,
        Timings,
    )

    # Check attn output
    # cosine similarity flattens error across heads and head_dim
    cosine_similarity = torch.nn.functional.cosine_similarity(
        O.float().flatten(start_dim=-2, end_dim=-1),
        ref.float().flatten(start_dim=-2, end_dim=-1),
        dim=-1,
    )
    is_correct = cosine_similarity > 0.99
    if not is_correct.all() and cosine_similarity.min() < 0.9:
        print(
            f"Cosine similarity mean: {cosine_similarity.mean()}, worst: {cosine_similarity.min()}"
        )
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

    # Check kv update (last N tokens match between K_cache and K_new)
    for b in range(len(Lengths)):
        # find page where the end of sequence lands
        eos_page_idx = Lengths[b] // PAGE_SIZE
        eos_page_addr = Table[b, eos_page_idx]
        offset_in_page = Lengths[b] % PAGE_SIZE

        space_in_eos_page = PAGE_SIZE - offset_in_page
        tokens_in_eos_page = min(new_tokens, space_in_eos_page)

        cached_K = K_cache[
            eos_page_addr, offset_in_page : offset_in_page + tokens_in_eos_page
        ]
        cached_V = V_cache[
            eos_page_addr, offset_in_page : offset_in_page + tokens_in_eos_page
        ]

        if new_tokens > tokens_in_eos_page:
            spillover_tokens = new_tokens - tokens_in_eos_page
            next_page_addr = Table[b, eos_page_idx + 1]
            extra_K = K_cache[next_page_addr, :spillover_tokens]
            extra_V = V_cache[next_page_addr, :spillover_tokens]
            cached_K = torch.cat([cached_K, extra_K], dim=-2)
            cached_V = torch.cat([cached_V, extra_V], dim=-2)

        new_K = K_new_rope_applied[b, :]
        new_V = V_new[b, :]
        if not torch.allclose(cached_K, new_K, atol=1e-1):
            print("Reference", new_K.view(-1, 8, 8))
            print("Candidate", cached_K.view(-1, 8, 8))
            print("Difference", (new_K - cached_K).view(-1, 8, 8))
            assert False, "K_cache update failed"
        if not torch.allclose(cached_V, new_V, atol=1e-3):
            print("Reference", new_V[..., :4])
            print("Candidate", cached_V[..., :4])
            assert False, "V_cache update failed"

    # time_per_iter = profile_thundergqa(
    #     Q,
    #     K_cache,
    #     V_cache,
    #     K_new,
    #     V_new,
    #     sin,
    #     cos,
    #     Table,
    #     Instructions,
    #     O_scratch,
    #     Lvec_scratch,
    #     Semaphore,
    #     Timings,
    # )
    # print(f"Time per iter: {time_per_iter * 1000} ms")

    # save_gantt_chart(Timings, Instructions, name="new")


if __name__ == "__main__":
    main([1], 1, 8)
    main([16], 4, 8)
    main([64], 2, 8)
    main([4641, 45118, 1730, 1696], 4, 8)
    main([65536], 1, 8)
    main([512] * 64, 2, 8)
    main([4096] * 132, 4, 8)
    main(
        [
            871,
            568,
            711,
            329,
            617,
            1015,
            348,
            978,
            543,
            837,
            650,
            1020,
            924,
            679,
            560,
            497,
            650,
            406,
            381,
            423,
            511,
            423,
            569,
            943,
            645,
            820,
            829,
            883,
            937,
            765,
            711,
            847,
            722,
            546,
            519,
            279,
            516,
            315,
            664,
            845,
            850,
            546,
            670,
            871,
            527,
            329,
            446,
            764,
            582,
            1011,
            453,
            655,
            532,
            985,
            1019,
            810,
            317,
            305,
            949,
            317,
            669,
            768,
            530,
            349,
        ],
        4,
        8,
    )
