# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op


@triton.jit
def _gated_norm_head(
    p_head,
    p_gate_head,
    norm_weight,
    norm_eps,
    out_fp4,
    out_scale,
    i_row,
    V: tl.constexpr,
    GATE_SIGMOID: tl.constexpr,
    QUANT_MXFP4: tl.constexpr,
):
    """rmsnorm(o) * w * act(z) over one value head, read back from the
    rounded output as the unfused norm reads it. Writes the head in place,
    or with QUANT_MXFP4 its MXFP4 bytes and e8m0 scales (the activation is
    rounded to the output dtype first, as before a separate quant)."""
    offs = tl.arange(0, V)
    # Written by other CUs: read through L2, not this CU's L1.
    x = tl.load(p_head + offs, cache_modifier=".cg").to(tl.float32)
    z = tl.load(p_gate_head + offs).to(tl.float32)
    w = tl.load(norm_weight + offs).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x) / V + norm_eps) * w
    if GATE_SIGMOID:
        y = y * tl.sigmoid(z)
    else:
        y = y * z * tl.sigmoid(z)
    y = y.to(p_head.dtype.element_ty)
    if QUANT_MXFP4:
        y_fp4, y_scale = _mxfp4_quant_op(y.to(tl.float32)[None, :], V, 1, 32)
        tl.store(out_fp4 + i_row * (V // 2) + tl.arange(0, V // 2)[None, :], y_fp4)
        tl.store(
            out_scale + i_row * (V // 32) + tl.arange(0, V // 32)[None, :],
            y_scale.to(out_scale.dtype.element_ty),
        )
    else:
        tl.store(p_head + offs, y)


@triton.jit
def _wait_vmem(x):
    """s_waitcnt vmcnt(0) in every wave: a workgroup-scope release does not
    wait for global stores (all its observers share the CU's L1), but the
    epilogue's reader runs on another CU of the same XCD."""
    return tl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)\n v_mov_b32 $0, $1",
        "=v,v",
        [x],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
        "FUSE_GATED_NORM": lambda args: args["norm_weight"] is not None,
        "QUANT_MXFP4": lambda args: args["out_fp4"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_rearrange_sigmoid_gated_delta_rule_update_kernel(
    A_log,
    a,
    b,
    dt_bias,
    beta,
    threshold,
    qkv,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    norm_weight,
    gate,
    norm_eps,
    norm_counter,
    out_fp4,
    out_scale,
    N: tl.int64,  # num of sequences
    T: tl.int64,  # num of tokens
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_qkv_l: tl.constexpr,
    stride_qkv_hd: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    stride_gate_tok: tl.constexpr,
    stride_gate_head: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
    FUSE_GATED_NORM: tl.constexpr,
    GATE_SIGMOID: tl.constexpr,
    QUANT_MXFP4: tl.constexpr,
    XCD_LOCAL: tl.constexpr,
):
    if XCD_LOCAL:
        # Workgroups are dispatched round-robin over (up to) 8 XCDs, each
        # with its own L2. Give the NV programs of one (token, head) ids
        # that are equal mod 8 so they share an L2, which lets the gated-norm
        # handshake below stay at workgroup scope; a device-scope
        # release/acquire costs every program an L2 writeback + invalidate.
        NV: tl.constexpr = V // BV
        pid = tl.program_id(1) + NV * tl.program_id(2)
        j = pid % (NV * 8)
        i_k, i_v, i_nh = 0, j // 8, (pid // (NV * 8)) * 8 + j % 8
    else:
        i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = qkv + bos * stride_qkv_l + ((i_h * K) + o_k) * stride_qkv_hd
    p_k = qkv + bos * stride_qkv_l + (H * K + (i_h * K) + o_k) * stride_qkv_hd
    p_v = qkv + bos * stride_qkv_l + (2 * H * K + (i_hv * V) + o_v) * stride_qkv_hd

    p_A_log = A_log + i_hv
    if not IS_KDA:
        p_a = a + bos * HV + i_hv
        p_dt_bias = dt_bias + i_hv
    else:
        p_a = a + (bos * HV + i_hv) * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k

    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            state_idx = tl.load(
                ssm_state_indices + i_n * stride_indices_seq + i_t * stride_indices_tok
            ).to(tl.int64)
            if state_idx < 0:
                return
            p_h0 = h0 + state_idx * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * V * K
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        x = tl.load(p_a).to(tl.float32) + tl.load(p_dt_bias).to(tl.float32)
        softplus_x = tl.where(
            beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
        )
        b_g = -tl.exp(tl.load(p_A_log).to(tl.float32)) * softplus_x

        b_beta = tl.sigmoid(b_b.to(tl.float32))

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        if not IS_KDA:
            b_h *= tl.exp(b_g)
        else:
            b_h *= tl.exp(b_g[None, :])
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        if FUSE_GATED_NORM:
            # The head's V outputs are split over the NV programs of this
            # (token, head). Each counts in after its store; the last one
            # in normalizes the head and resets the counter for the next
            # launch. No program waits, so residency does not matter.
            i_tok = bos + i_t
            p_cnt = norm_counter + i_tok * HV + i_hv
            if XCD_LOCAL:
                _wait_vmem(o_v)
                n_in = tl.atomic_add(p_cnt, 1, sem="acq_rel", scope="cta")
            else:
                n_in = tl.atomic_add(p_cnt, 1, sem="acq_rel", scope="gpu")
            if n_in == tl.cdiv(V, BV) - 1:
                _gated_norm_head(
                    o + ((i_k * all + i_tok) * HV + i_hv) * V,
                    gate + i_tok * stride_gate_tok + i_hv * stride_gate_head,
                    norm_weight,
                    norm_eps,
                    out_fp4,
                    out_scale,
                    i_tok * HV + i_hv,
                    V,
                    GATE_SIGMOID,
                    QUANT_MXFP4,
                )
                if XCD_LOCAL:
                    tl.atomic_xchg(p_cnt, 0, sem="relaxed", scope="cta")
                else:
                    tl.atomic_xchg(p_cnt, 0, sem="relaxed", scope="gpu")

        if INPLACE_FINAL_STATE:
            final_state_idx = tl.load(
                ssm_state_indices + i_n * stride_indices_seq + i_t * stride_indices_tok
            ).to(tl.int64)
            if final_state_idx >= 0:
                p_ht = ht + final_state_idx * stride_final_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += stride_qkv_l
        p_k += stride_qkv_l
        p_v += stride_qkv_l
        p_o += HV * V
        p_b += HV
        p_a += HV
