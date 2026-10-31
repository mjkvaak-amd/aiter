# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.gemm.fused.fused_gemm_argmax import (
    _argmax_pair_pick_kernel,
    _argmax_partial_reduce_kernel,
    _fused_gemm_argmax_kernel,
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

_REDUCE_BLOCK_SIZE = 1024


def _gemm_argmax(x, w, w_scale, n_valid, index_offset, out, config):
    M, K = x.shape
    N, Kw = w.shape
    assert K == Kw, "Incompatible matrix shapes."
    assert x.dtype in (torch.bfloat16, torch.float16), "x must be BF16 or FP16."
    assert N + index_offset <= 2**24, "Indices must fit exactly in FP32."
    if n_valid is None:
        n_valid = N

    if config is None:
        config, _ = _get_config(M, N, K, w.element_size() * 8)
    num_partials = triton.cdiv(N, config["BLOCK_SIZE_N"])

    vals = torch.empty((M, num_partials), dtype=torch.float32, device=x.device)
    idxs = torch.empty((M, num_partials), dtype=torch.int32, device=x.device)
    if out is None:
        out = torch.empty((M, 2), dtype=torch.float32, device=x.device)

    grid = (num_partials, triton.cdiv(M, config["BLOCK_SIZE_M"]))
    _fused_gemm_argmax_kernel[grid](
        x,
        w,
        w_scale,
        vals,
        idxs,
        M,
        N,
        n_valid,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        vals.stride(0),
        HAS_W_SCALE=w_scale is not None,
        **config,
    )
    _argmax_partial_reduce_kernel[(M,)](
        vals,
        idxs,
        out,
        num_partials,
        vals.stride(0),
        out.stride(0),
        index_offset,
        BLOCK_SIZE=min(_REDUCE_BLOCK_SIZE, triton.next_power_of_2(num_partials)),
    )
    return out


def fused_gemm_a16w16_argmax(
    x: torch.Tensor,
    w: torch.Tensor,
    n_valid: int | None = None,
    index_offset: int = 0,
    out: torch.Tensor | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    """
    Computes the row argmax of the logits X @ W^T without storing the logits.

    The kernel rounds each logit to the dtype of X before it compares them.
    A tie selects the smallest index. The result is the same as
    ``(x @ w.T).max(dim=-1)`` on 16-bit logits.

    Args:
        x (torch.Tensor): Input with shape (M, K), BF16 or FP16.
        w (torch.Tensor): Weight with shape (N, K) and the dtype of x. Any
            strides are accepted.
        n_valid (Optional[int]): Number of valid weight rows. Rows at and after
            n_valid are ignored. Default is N.
        index_offset (int): Value added to each index, for example the first
            vocabulary index of a tensor-parallel shard.
        out (Optional[torch.Tensor]): Preallocated FP32 output with shape (M, 2).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M,
            BLOCK_SIZE_N, BLOCK_SIZE_K, num_warps, num_stages, waves_per_eu,
            matrix_instr_nonkdim, cache_modifier).

    Returns:
        torch.Tensor: FP32 tensor with shape (M, 2). Column 0 is the maximum
        logit and column 1 is the index of the maximum plus index_offset. FP32
        holds indices up to 2**24 exactly.
    """
    _LOGGER.info("FUSED_GEMM_A16W16_ARGMAX: x=%s w=%s", tuple(x.shape), tuple(w.shape))
    assert x.dtype == w.dtype, "x and w must have the same dtype."
    return _gemm_argmax(x, w, None, n_valid, index_offset, out, config)


def fused_gemm_a16w8_argmax(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    n_valid: int | None = None,
    index_offset: int = 0,
    out: torch.Tensor | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    """
    Computes the row argmax of the logits X @ (W * w_scale)^T without storing
    the logits. W is FP8 with one scale for each row.

    X stays in 16 bits. A per-token scale of X does not change the argmax, so
    this op also gives the argmax of an FP8 GEMM with per-token activation
    scales, but without the activation quantization error. The kernel rounds
    each logit to the dtype of X before it compares them. A tie selects the
    smallest index.

    Args:
        x (torch.Tensor): Input with shape (M, K), BF16 or FP16.
        w (torch.Tensor): FP8 weight with shape (N, K). Any strides are accepted.
        w_scale (torch.Tensor): FP32 scales, N contiguous values.
        n_valid (Optional[int]): Number of valid weight rows. Default is N.
        index_offset (int): Value added to each index.
        out (Optional[torch.Tensor]): Preallocated FP32 output with shape (M, 2).
        config (Optional[dict]): Kernel tuning parameters, as in
            ``fused_gemm_a16w16_argmax``.

    Returns:
        torch.Tensor: FP32 tensor with shape (M, 2), as in
        ``fused_gemm_a16w16_argmax``.
    """
    _LOGGER.info("FUSED_GEMM_A16W8_ARGMAX: x=%s w=%s", tuple(x.shape), tuple(w.shape))
    assert w.element_size() == 1, "w must be FP8."
    assert w_scale.dtype == torch.float32 and w_scale.numel() == w.shape[0]
    assert w_scale.is_contiguous(), "w_scale must be contiguous."
    return _gemm_argmax(x, w, w_scale, n_valid, index_offset, out, config)


def argmax_pair_pick(
    pairs: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Selects the index of the largest of T (value, index) pairs in each row.

    Use it after an all-gather of the outputs of ``fused_gemm_a16w16_argmax``
    or ``fused_gemm_a16w8_argmax`` from T tensor-parallel ranks. A tie selects
    the first pair.

    Args:
        pairs (torch.Tensor): FP32 tensor with shape (M, 2 * T) and stride 1
            in the last dimension. Pair t is at columns 2 * t and 2 * t + 1.
        out (Optional[torch.Tensor]): Preallocated int64 output with shape (M,).

    Returns:
        torch.Tensor: int64 tensor with shape (M,).
    """
    M, width = pairs.shape
    assert width % 2 == 0, "pairs must have shape (M, 2 * T)."
    assert pairs.dtype == torch.float32 and pairs.stride(1) == 1
    T = width // 2
    if out is None:
        out = torch.empty((M,), dtype=torch.int64, device=pairs.device)
    _argmax_pair_pick_kernel[(M,)](
        pairs,
        out,
        T,
        pairs.stride(0),
        BLOCK_SIZE_T=triton.next_power_of_2(T),
    )
    return out
