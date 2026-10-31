# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.gemm.fused.fused_gemm_mtp_residual_linear_shared import (
    _fused_gemm_mtp_residual_linear_shared_kernel,
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def fused_gemm_mtp_residual_linear_shared(
    e: torch.Tensor,
    h: torch.Tensor,
    e_norm_weight: torch.Tensor,
    h_norm_weight: torch.Tensor,
    w_e: torch.Tensor,
    w_h: torch.Tensor,
    eps: float,
    out: torch.Tensor | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    """
    Computes the input projection of an MTP layer with a shared hidden
    projection and the embedding as a residual of each stream:

        out[t, c] = gemma_rmsnorm(h[t])[c] @ w_h^T + gemma_rmsnorm(e[t]) @ w_e^T

    A Gemma RMSNorm scales by (1 + weight). The norm of h spans all C streams
    of a token. The normalized inputs are not stored.

    Args:
        e (torch.Tensor): Token embeddings with shape (T, K), BF16 or FP16.
        h (torch.Tensor): Hidden states with shape (T, C * K), C streams of K.
            C must be a power of two.
        e_norm_weight (torch.Tensor): Norm weight of e with shape (K,).
        h_norm_weight (torch.Tensor): Norm weight of h with shape (C * K,).
        w_e (torch.Tensor): Embedding projection with shape (N, K).
        w_h (torch.Tensor): Hidden projection with shape (N, K), shared by
            all streams.
        eps (float): Epsilon of both norms.
        out (Optional[torch.Tensor]): Preallocated output with shape (T, C * N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_T,
            BLOCK_SIZE_N, BLOCK_SIZE_K, num_warps, num_stages, waves_per_eu,
            matrix_instr_nonkdim). BLOCK_SIZE_T is raised to 16 // C.

    Returns:
        torch.Tensor: Output with shape (T, C * N) and the dtype of e.
    """
    _LOGGER.info(
        "FUSED_GEMM_MTP_RESIDUAL_LINEAR_SHARED: e=%s h=%s w=%s",
        tuple(e.shape),
        tuple(h.shape),
        tuple(w_h.shape),
    )
    T, K = e.shape
    N = w_h.shape[0]
    C = h.shape[1] // K
    assert h.shape == (T, C * K) and C & (C - 1) == 0, "h must be (T, C * K)."
    assert w_e.shape == (N, K) and w_h.shape == (N, K)
    assert e_norm_weight.shape == (K,) and h_norm_weight.shape == (C * K,)
    for t in (e, h, w_e, w_h, e_norm_weight, h_norm_weight):
        assert t.stride(-1) == 1, "The last dimension must be contiguous."
    if out is None:
        out = torch.empty((T, C * N), dtype=e.dtype, device=e.device)

    if config is None:
        config, _ = _get_config(T, N, K)
    # tl.dot needs 16 rows: BLOCK_SIZE_T tokens of C streams each.
    config = {**config, "BLOCK_SIZE_T": max(config["BLOCK_SIZE_T"], 16 // C)}

    grid = (
        triton.cdiv(N, config["BLOCK_SIZE_N"]),
        triton.cdiv(T, config["BLOCK_SIZE_T"]),
    )
    _fused_gemm_mtp_residual_linear_shared_kernel[grid](
        e,
        h,
        e_norm_weight,
        h_norm_weight,
        w_e,
        w_h,
        out,
        T,
        N,
        K,
        e.stride(0),
        h.stride(0),
        w_e.stride(0),
        w_h.stride(0),
        out.stride(0),
        eps,
        C=C,
        **config,
    )
    return out
