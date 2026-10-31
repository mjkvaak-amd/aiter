# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

_INT32_MAX = tl.constexpr(2147483647)

_fused_gemm_argmax_repr = make_kernel_repr(
    "_fused_gemm_argmax_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "EVEN_K",
        "HAS_W_SCALE",
        "cache_modifier",
    ],
)

_argmax_partial_reduce_repr = make_kernel_repr(
    "_argmax_partial_reduce_kernel",
    ["BLOCK_SIZE"],
)

_argmax_pair_pick_repr = make_kernel_repr(
    "_argmax_pair_pick_kernel",
    ["BLOCK_SIZE_T"],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit(repr=_fused_gemm_argmax_repr)
def _fused_gemm_argmax_kernel(
    x_ptr,
    w_ptr,
    w_scale_ptr,
    val_ptr,
    idx_ptr,
    M,
    N,
    N_valid,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_pm,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    HAS_W_SCALE: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """Computes one (BLOCK_SIZE_M, BLOCK_SIZE_N) tile of X @ W^T and stores
    the max and the argmax of each row of the tile. With HAS_W_SCALE, W is FP8
    and row n of W is scaled by w_scale[n]."""
    tl.assume(stride_xm > 0)
    tl.assume(stride_xk > 0)
    tl.assume(stride_wn > 0)
    tl.assume(stride_wk > 0)

    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
            w = tl.load(
                w_ptrs, mask=mask_n[None, :], other=0.0, cache_modifier=cache_modifier
            )
        else:
            k_left = K - k * BLOCK_SIZE_K
            x = tl.load(
                x_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < k_left), other=0.0
            )
            w = tl.load(
                w_ptrs,
                mask=mask_n[None, :] & (offs_k[:, None] < k_left),
                other=0.0,
                cache_modifier=cache_modifier,
            )
        if HAS_W_SCALE:
            w = w.to(x_ptr.type.element_ty)
        acc = tl.dot(x, w, acc=acc)
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    if HAS_W_SCALE:
        w_scale = tl.load(w_scale_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc * w_scale[None, :]
    # Compare the logits at the output precision, so that ties resolve like
    # an argmax over the stored logits.
    logits = acc.to(x_ptr.type.element_ty).to(tl.float32)
    logits = tl.where((offs_n < N_valid)[None, :], logits, float("-inf"))
    best = tl.max(logits, axis=1)
    best_n = tl.argmax(logits, axis=1, tie_break_left=True)

    p_offs = offs_m * stride_pm + pid_n
    tl.store(val_ptr + p_offs, best, mask=mask_m)
    tl.store(idx_ptr + p_offs, pid_n * BLOCK_SIZE_N + best_n, mask=mask_m)


@triton.jit(repr=_argmax_partial_reduce_repr)
def _argmax_partial_reduce_kernel(
    val_ptr,
    idx_ptr,
    out_ptr,
    num_partials,
    stride_pm,
    stride_om,
    index_offset,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduces the tile maxima of one row to a (value, index) pair. The index
    is stored as float32, with index_offset added."""
    pid_m = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    best = tl.full((), float("-inf"), tl.float32)
    best_i = tl.full((), _INT32_MAX, tl.int32)
    for start in tl.range(0, num_partials, BLOCK_SIZE):
        mask = start + offs < num_partials
        v = tl.load(
            val_ptr + pid_m * stride_pm + start + offs, mask=mask, other=float("-inf")
        )
        i = tl.load(
            idx_ptr + pid_m * stride_pm + start + offs, mask=mask, other=_INT32_MAX
        )
        v_max = tl.max(v, axis=0)
        i_min = tl.min(tl.where(v == v_max, i, _INT32_MAX), axis=0)
        # Later chunks hold larger indices, so a tie keeps the current best.
        take = v_max > best
        best_i = tl.where(take, i_min, best_i)
        best = tl.where(take, v_max, best)
    best_i = tl.where(best_i == _INT32_MAX, 0, best_i)
    tl.store(out_ptr + pid_m * stride_om, best)
    tl.store(out_ptr + pid_m * stride_om + 1, (best_i + index_offset).to(tl.float32))


@triton.jit(repr=_argmax_pair_pick_repr)
def _argmax_pair_pick_kernel(
    pair_ptr,
    out_ptr,
    T,
    stride_pm,
    BLOCK_SIZE_T: tl.constexpr,
):
    """Picks the index of the largest of T (value, index) pairs of one row.
    A tie picks the first pair."""
    pid_m = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE_T)
    mask = offs < T
    v = tl.load(pair_ptr + pid_m * stride_pm + 2 * offs, mask=mask, other=float("-inf"))
    i = tl.load(pair_ptr + pid_m * stride_pm + 2 * offs + 1, mask=mask, other=0.0)
    first = tl.argmax(v, axis=0, tie_break_left=True)
    pick = tl.sum(tl.where(offs == first, i, 0.0), axis=0)
    tl.store(out_ptr + pid_m, pick.to(tl.int64))


def _get_config(M: int, N: int, K: int, w_bits: int = 16):
    name = "FUSED-GEMM-A16W16-ARGMAX" if w_bits == 16 else "FUSED-GEMM-A16W8-ARGMAX"
    return get_gemm_config(name, M, N, K)
