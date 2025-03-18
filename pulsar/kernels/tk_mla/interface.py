from __future__ import annotations

import torch

from typing import Tuple

from tk_mla.mla_decode import mla_decode, mla_decode_8_heads
from tk_mla.scheduler import create_thundermla_arguments

NUM_PROCESSORS: int = 132


def get_schedules(
    cache_seqlens: torch.Tensor,
    decode_seqlen: int,
    num_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    instructions, output_scratch, intermediate_scratch, semaphore, _ = create_thundermla_arguments(
        cache_seqlens,
        decode_seqlen,
        num_heads,
    )

    return instructions, output_scratch, intermediate_scratch, semaphore


def decode(
    # The Q component.
    query: torch.Tensor,
    # The cached KV components.
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    # The KV components.
    key: torch.Tensor,
    value: torch.Tensor,
    # Qv
    query_v: torch.Tensor,
    # The Q and new K cumulative sequence lengths.
    cu_seqlen_query: torch.Tensor,
    cu_seqlen_key: torch.Tensor,
    # The sequence length bounds for K.
    cache_seqlens: torch.Tensor | None,
    # The block table for paged-kv.
    block_table: torch.Tensor | None,
    # The KV batch index.
    cache_indices: torch.Tensor | None,
    # The left padding for Q and K.
    prefix_seqlens: torch.Tensor | None,
    # The cached rotary embedding supports.
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    # Optional scales for query, key and value.
    scales_q: torch.Tensor | None,
    scales_k: torch.Tensor | None,
    scales_v: torch.Tensor | None,
    # The softmax scale in attention.
    softmax_scale: float,
    # Causal self-attention.
    causal: bool = True,
    # Sliding window attention specific(s).
    left_window: int | None = None,
    right_window: int | None = None,
    # Attention logits soft-capping.
    softcap: float | None = None,
    # Non-interleaved rotary embedding.
    interleaved: bool = True,
    # The schedule for ThunderMLA.
    instructions: torch.Tensor | None = None,
    output_scratch: torch.Tensor | None = None,
    intermediate_scratch: torch.Tensor | None = None,
    semaphore: torch.Tensor | None = None,
) -> torch.Tensor:
    
    if key is not None or value is not None:

        raise NotImplementedError("Fused KV Cache Update is not supported in ThunderMLA.")
    
    if cu_seqlen_query is not None or cu_seqlen_key is not None:

        raise NotImplementedError("Varlen is not supported in ThunderMLA.")
    
    if not causal:

        raise NotImplementedError("Non-Causal attention is not supported in ThunderMLA.")
    
    if scales_q is not None or scales_k is not None or scales_v is not None:

        raise NotImplementedError("FP8 attention is not supported in ThunderMLA.")
    
    if left_window is not None or right_window is not None:

        raise NotImplementedError("Sliding window attention is not supported in ThunderMLA.")
    
    if softcap is not None:

        raise NotImplementedError("Soft-capping is not supported in ThunderMLA.")
    
    if not interleaved:

        raise NotImplementedError("Non-interleaved rotary embedding is not supported in ThunderMLA.")
    
    if prefix_seqlens:
    
        raise NotImplementedError("Prefix sharing is not supported in ThunderMLA.")

    num_heads = query.size(2)

    output = torch.zeros_like(query_v)
    
    kernel = mla_decode_8_heads if num_heads == 8 else mla_decode
    
    kernel(
        instructions,
        query,
        query_v,
        sin,
        cos,
        kcache,
        vcache,
        block_table,
        output,
        output_scratch,
        intermediate_scratch,
        semaphore,
        softmax_scale,
        1,
    )

    return output
