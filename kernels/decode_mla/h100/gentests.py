from __future__ import annotations

import torch

from math import sqrt

from torch.nn import init
from torch.nn.functional import linear, scaled_dot_product_attention as sdpa

# from mla import flash_mla_decode

DTYPE: torch.dtype = torch.bfloat16
DEVICE: torch.device = torch.device('cuda:0')

def mitchell(w: torch.Tensor) -> torch.Tensor:

    sigma = 1 / sqrt(w.size(-1))
    bound = 3 * sigma

    init.trunc_normal_(w, mean=0.0, std=sigma, a=-bound, b=bound)


# QK Dimension(s).
B, R, H, Z, Zc, Zr = 4, 0, 16, 576, 512, 64

# Input Dimension(s).
L = B * (R + 1)
D = 7_168

# Softmax Scale.
Sz = 1 / sqrt(Z)

# Latent Cache Specific(s).
N, K = 128, 256

# Sequence Length Specific(s).
T = 32_768 // K
Sl, Sh = 32, 1_024


with torch.device(DEVICE):

    # Initialize Input(s) and Cache(s).

    hidden = torch.randn(L, D, dtype=DTYPE)
    cache = torch.empty(N, K, 1, Z, dtype=DTYPE)

    mitchell(weight := torch.empty(H * Z, D, dtype=DTYPE))

    query = (
        linear(hidden, weight)
        .view(B, R + 1, H, Z)
    )


    # Define Sequence Length and Block Size(s).

    lengths = torch.randint(Sl, Sh, (B,), dtype=torch.int32)
    sizes = (lengths + (K - 1)).floor_divide(K)

    total = lengths.sum().item()
    maximum = lengths.max().item()


    # Allocate Block Table.

    table = torch.zeros(B, T, dtype=torch.int32)

    sequence_ids, position_ids = (
        torch.arange(T, dtype=torch.int32)[None, :]
        .expand(B, -1)
        .lt(sizes.view(-1, 1))
        .nonzero(as_tuple=True)
    )

    table[sequence_ids, position_ids] = (
        torch.randperm(N)
        [:sequence_ids.size(0)]
        .sort()
        .values
        .int()
    )


    # Prefill Latent Cache & Extract Padded.

    latent = torch.randn(total, 1, Z, dtype=DTYPE)
    expanded = latent.expand(total, H, Z)

    sequence_ids, position_ids = (
        torch.arange(maximum)[None, :]
        .expand(B, -1)
        .lt(lengths.view(-1, 1))
        .nonzero(as_tuple=True)
    )

    padded_key = torch.zeros(B, maximum, H, Z, dtype=DTYPE)
    padded_value = torch.zeros(B, maximum, H, Zc, dtype=DTYPE)

    padded_key[sequence_ids, position_ids] = expanded
    padded_value[sequence_ids, position_ids] = expanded[..., :Zc]

    entry_ids = table[sequence_ids, position_ids.floor_divide(K)]
    column_ids = position_ids.fmod(K)

    cache[entry_ids, column_ids] = latent


    # Construct (Optional) Attention Mask.

    mask = None

    if R > 0:

        bounds = (
            torch.arange(R + 1, dtype=torch.int32)[None, :]
            + lengths[:, None]
            - R
        )

        mask = (
            torch.arange(maximum, dtype=torch.int32)[None, None, None, :]
            .expand(B, H, R + 1, -1)
            .lt(bounds[:, None, :, None].expand(B, H, R + 1, 1))
        )


# Execution Barrier.

torch.cuda.current_stream().synchronize()


def naive_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, scale: float) -> torch.Tensor:

    scores = torch.einsum(
        '... s z , ... t z -> ... s t',
        q.transpose(1, 2),
        k.transpose(1, 2),
    )

    out = torch.einsum(
        '... s t , ... t z -> ... s z',
        scores.mul_(scale).softmax(dim=-1),
        v.transpose(1, 2),
    )

    return out.transpose(1, 2)


# SDPA Kernel.

padded_o1 = sdpa(
    query=query.transpose(1, 2),
    key=padded_key.transpose(1, 2),
    value=padded_value.transpose(1, 2),
    attn_mask=mask,
    dropout_p=0.0,
    is_causal=False,
    scale=Sz,
    enable_gqa=False,
)

o1 = padded_o1.transpose(1, 2)


# Execution Barrier,

torch.cuda.current_stream().synchronize()


# Target Kernel.

# o2 = flash_mla_decode(
#     query=query,
#     latent_cache=cache,
#     cache_seqlens=lengths,
#     block_table=table,
#     softmax_scale=Sz,
#     latent_dim=Zc,
#     rope_dim=Zr,
#     naive_reduce=False,
# )


# Execution Barrier,

torch.cuda.current_stream().synchronize()


# Naive Torch.

o3 = naive_sdpa(query, padded_key, padded_value, scale=Sz)