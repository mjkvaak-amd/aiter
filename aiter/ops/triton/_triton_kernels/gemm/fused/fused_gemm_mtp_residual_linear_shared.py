# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

_fused_gemm_mtp_residual_linear_shared_repr = make_kernel_repr(
    "_fused_gemm_mtp_residual_linear_shared_kernel",
    ["C", "BLOCK_SIZE_T", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "EVEN_K"],
)


@triton.heuristics({"EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0})
@triton.jit(repr=_fused_gemm_mtp_residual_linear_shared_repr)
def _fused_gemm_mtp_residual_linear_shared_kernel(
    e_ptr,
    h_ptr,
    e_norm_w_ptr,
    h_norm_w_ptr,
    w_e_ptr,
    w_h_ptr,
    out_ptr,
    T,
    N,
    K,
    stride_et,
    stride_ht,
    stride_wen,
    stride_whn,
    stride_ot,
    eps,
    C: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """out[t, c] = rmsnorm(h[t])[c] @ W_h^T + rmsnorm(e[t]) @ W_e^T, with Gemma
    norms. The norm of h spans all C streams of a token. Each norm scale is
    applied after the dot, from sums of squares taken in the same K loop."""
    pid_n = tl.program_id(axis=0)
    pid_t = tl.program_id(axis=1)
    BLOCK_SIZE_R: tl.constexpr = BLOCK_SIZE_T * C

    offs_r = pid_t * BLOCK_SIZE_T * C + tl.arange(0, BLOCK_SIZE_R)
    row_t = offs_r // C
    row_c = offs_r % C
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    mask_r = row_t < T
    mask_n = offs_n < N

    # Rows are (token, stream) pairs. Each stream row also loads the token's e.
    e_ptrs = e_ptr + row_t[:, None] * stride_et + offs_k[None, :]
    h_ptrs = h_ptr + row_t[:, None] * stride_ht + (row_c * K)[:, None] + offs_k[None, :]
    hw_ptrs = h_norm_w_ptr + (row_c * K)[:, None] + offs_k[None, :]
    we_ptrs = w_e_ptr + offs_n[None, :] * stride_wen + offs_k[:, None]
    wh_ptrs = w_h_ptr + offs_n[None, :] * stride_whn + offs_k[:, None]

    acc_e = tl.zeros((BLOCK_SIZE_R, BLOCK_SIZE_N), dtype=tl.float32)
    acc_h = tl.zeros((BLOCK_SIZE_R, BLOCK_SIZE_N), dtype=tl.float32)
    sq_e = tl.zeros((BLOCK_SIZE_R,), dtype=tl.float32)
    sq_h = tl.zeros((BLOCK_SIZE_R,), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k0 = k * BLOCK_SIZE_K
        if EVEN_K:
            mask_k = offs_k < BLOCK_SIZE_K
        else:
            mask_k = offs_k < K - k0
        e = tl.load(e_ptrs, mask=mask_r[:, None] & mask_k[None, :], other=0.0)
        h = tl.load(h_ptrs, mask=mask_r[:, None] & mask_k[None, :], other=0.0)
        e_nw = tl.load(e_norm_w_ptr + k0 + offs_k, mask=mask_k, other=0.0)
        h_nw = tl.load(hw_ptrs, mask=mask_k[None, :], other=0.0)
        w_e = tl.load(we_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        w_h = tl.load(wh_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        e32 = e.to(tl.float32)
        h32 = h.to(tl.float32)
        sq_e += tl.sum(e32 * e32, axis=1)
        sq_h += tl.sum(h32 * h32, axis=1)
        e_in = (e32 * (1.0 + e_nw.to(tl.float32))[None, :]).to(w_e.dtype)
        h_in = (h32 * (1.0 + h_nw.to(tl.float32))).to(w_h.dtype)
        acc_e = tl.dot(e_in, w_e, acc=acc_e)
        acc_h = tl.dot(h_in, w_h, acc=acc_h)

        e_ptrs += BLOCK_SIZE_K
        h_ptrs += BLOCK_SIZE_K
        hw_ptrs += BLOCK_SIZE_K
        we_ptrs += BLOCK_SIZE_K
        wh_ptrs += BLOCK_SIZE_K

    rr_e = tl.rsqrt(sq_e / K + eps)
    sq_tok = tl.sum(tl.reshape(sq_h, (BLOCK_SIZE_T, C)), axis=1)
    rr_tok = tl.rsqrt(sq_tok / (K * C) + eps)
    rr_h = tl.reshape(
        tl.broadcast_to(rr_tok[:, None], (BLOCK_SIZE_T, C)), (BLOCK_SIZE_R,)
    )
    out = acc_h * rr_h[:, None] + acc_e * rr_e[:, None]

    out_ptrs = (
        out_ptr + row_t[:, None] * stride_ot + (row_c * N)[:, None] + offs_n[None, :]
    )
    tl.store(
        out_ptrs,
        out.to(out_ptr.type.element_ty),
        mask=mask_r[:, None] & mask_n[None, :],
    )


def _get_config(T: int, N: int, K: int):
    return get_gemm_config("FUSED-GEMM-MTP-RESIDUAL-LINEAR-SHARED", T, N, K)
