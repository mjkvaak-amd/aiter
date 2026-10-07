# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Adapted from flash-linear-attention / vLLM (see _triton_kernels copy).

from __future__ import annotations

import os

import torch
import triton

from aiter.ops.triton._triton_kernels.gated_delta_net.decode.fused_rearrange_sigmoid_gdr import (
    fused_rearrange_sigmoid_gated_delta_rule_update_kernel,
)


def _flydsl_gdr_enabled() -> bool:
    """Opt-in gate for the FlyDSL gated-delta-rule MTP port.

    Off by default, so Triton keeps serving every call. Not memoized, so the env
    var can be toggled at runtime.
    """
    return os.environ.get("AITER_GDR_FLYDSL", "") == "1"


def _try_flydsl_mtp(
    *,
    A_log,
    a,
    b,
    dt_bias,
    qkv,
    key_dim,
    value_dim,
    head_k_dim,
    head_v_dim,
    softplus_beta,
    softplus_threshold,
    scale,
    initial_state,
    inplace_final_state,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    use_qk_l2norm_in_kernel,
    is_kda,
    core_attn_out,
    draft_window,
):
    """Route a speculative-verify call to the FlyDSL chain kernel, or decline.

    Returns ``(out, final_state)`` when it took the call and ``None`` when the
    caller should fall through to Triton. Declining is the common case: this
    covers the MTP verify shape and nothing else.
    """
    if is_kda or not inplace_final_state or initial_state is None:
        return None
    if num_accepted_tokens is None or ssm_state_indices is None:
        return None
    if ssm_state_indices.ndim != 2:
        return None
    if qkv.ndim != 2 or qkv.stride(1) != 1:
        return None
    # What `flydsl_gdr_mtp` asserts rather than screens, and the support
    # predicate only covers q/k/v and the state. Declining here is what keeps
    # the env var from turning a call Triton would have served into an
    # exception out of this entry point.
    if A_log.dtype not in (torch.float32, torch.bfloat16):
        return None
    if any(t.device != qkv.device for t in (a, b, dt_bias, A_log, initial_state)):
        return None
    # The state vector is loaded 16 bytes at a time.
    if initial_state.data_ptr() % 16 != 0:
        return None
    # The launch writes `out` at the operands' dtype, while the Triton path
    # writes into whatever it is handed.
    if core_attn_out is not None and (
        core_attn_out.dtype != qkv.dtype
        or core_attn_out.device != qkv.device
        or not core_attn_out.is_contiguous()
    ):
        return None
    # The gating constants are compiled into the kernel, so only the default
    # pair is routed.
    if float(softplus_beta) != 1.0 or float(softplus_threshold) != 20.0:
        return None
    if scale is not None and abs(float(scale) - head_k_dim**-0.5) > 1e-12:
        return None
    if qkv.is_cuda:
        with torch.cuda.device(qkv.device):
            if torch.cuda.is_current_stream_capturing():
                return None

    if not isinstance(draft_window, int) or draft_window <= 0 or cu_seqlens is None:
        return None
    total_tokens = qkv.shape[0]
    n_seq = ssm_state_indices.shape[0]
    if (
        ssm_state_indices.shape[1] != draft_window
        or n_seq * draft_window != total_tokens
        or cu_seqlens.numel() != n_seq + 1
    ):
        return None
    window = draft_window

    H = key_dim // head_k_dim
    HV = value_dim // head_v_dim
    if (
        core_attn_out is not None
        and core_attn_out.numel() < total_tokens * HV * head_v_dim
    ):
        return None
    stride_qkv_l = qkv.stride(0)
    base = qkv.storage_offset()

    # q / k / v are strided views into the packed projection rather than copies;
    # the kernel takes their strides as build parameters.
    q = qkv.as_strided(
        (n_seq, window, H, head_k_dim),
        (window * stride_qkv_l, stride_qkv_l, head_k_dim, 1),
        base,
    )
    k = qkv.as_strided(
        (n_seq, window, H, head_k_dim),
        (window * stride_qkv_l, stride_qkv_l, head_k_dim, 1),
        base + key_dim,
    )
    v = qkv.as_strided(
        (n_seq, window, HV, head_v_dim),
        (window * stride_qkv_l, stride_qkv_l, head_v_dim, 1),
        base + 2 * key_dim,
    )

    if a.ndim != 2 or b.ndim != 2 or a.stride(1) != 1 or b.stride(1) != 1:
        return None
    a_view = a.as_strided(
        (n_seq, window, HV), (window * a.stride(0), a.stride(0), 1), a.storage_offset()
    )
    b_view = b.as_strided(
        (n_seq, window, HV), (window * b.stride(0), b.stride(0), 1), b.storage_offset()
    )

    from aiter.ops.flydsl.linear_attention_kernels import (
        _flydsl_gdr_mtp_supported,
        _launch_flydsl_gdr_mtp,
    )

    idx = ssm_state_indices
    nacc = num_accepted_tokens
    if not _flydsl_gdr_mtp_supported(q, k, v, initial_state, idx, nacc):
        return None
    if a_view.dtype != qkv.dtype or b_view.dtype != qkv.dtype:
        return None
    if dt_bias.dtype != qkv.dtype:
        return None

    out = (
        core_attn_out.view(-1)[: total_tokens * HV * head_v_dim].view(
            n_seq, window, HV, head_v_dim
        )
        if core_attn_out is not None
        else qkv.new_empty(n_seq, window, HV, head_v_dim)
    )
    _launch_flydsl_gdr_mtp(
        query=q,
        key=k,
        value=v,
        a=a_view,
        b=b_view,
        dt_bias=dt_bias,
        A_log=A_log,
        state=initial_state,
        out=out,
        ssm_state_indices=idx,
        num_accepted_tokens=nacc,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        min_live_slot=0,
    )
    # Same rank as the Triton path below, which returns [1, T, HV, V].
    return out.view(1, total_tokens, HV, head_v_dim), initial_state


def fused_rearrange_sigmoid_gated_delta_rule(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    qkv: torch.Tensor,
    key_dim: int,
    value_dim: int,
    head_k_dim: int,
    head_v_dim: int,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
    core_attn_out: torch.Tensor | None = None,
    draft_window: int | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    gate: torch.Tensor | None = None,
    gate_activation: str = "silu",
    norm_counter: torch.Tensor | None = None,
    out_fp4: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused sigmoid-gated delta rule over packed QKV.

    ``draft_window`` is host metadata required for FlyDSL dispatch. FlyDSL
    falls back to Triton during CUDA Graph capture.

    With ``norm_weight`` the output is ``rmsnorm(o) * norm_weight * act(gate)``
    per value head (the Qwen3-Next / Qwen3.5 gated RMSNorm with
    ``norm_before_gate=True``), where ``gate`` is ``[T, HV, V]`` or
    ``[T, HV * V]`` and ``act`` is ``"silu"`` or ``"sigmoid"``.

    ``norm_counter`` is int32 scratch of at least ``T * HV`` elements that
    must be zero before the first call; every call leaves it zero. Without
    it, each call allocates and zeroes one, so pass a persistent buffer when
    the launch overhead matters.

    With ``out_fp4`` (uint8 ``[T, HV * V // 2]``) and ``out_scale`` (uint8
    ``[T, HV * V // 32]``), the normalized output is instead quantized to
    MXFP4 with unshuffled per-1x32 e8m0 scales, the layout
    ``fused_rms_gated_mxfp4_quant`` writes, and the returned output holds the
    raw, un-normalized ``o``.
    """
    # Spelled as raised ``AssertionError``s rather than ``assert`` statements,
    # keeping the type a caller may already handle while ``python -O`` can no
    # longer strip the guard.
    expected_shape = (qkv.shape[0], key_dim * 2 + value_dim)
    if qkv.shape != expected_shape:
        raise AssertionError(
            f"expect qkv to be in shape {expected_shape}, got {qkv.shape}"
        )
    # Both paths get their head counts by floor-dividing these, so a remainder
    # silently narrows every view by its width and returns a result shaped for
    # the heads that survived, rather than saying the layout does not divide.
    if key_dim % head_k_dim != 0:
        raise AssertionError(
            f"key_dim {key_dim} must be a multiple of head_k_dim {head_k_dim}"
        )
    if value_dim % head_v_dim != 0:
        raise AssertionError(
            f"value_dim {value_dim} must be a multiple of head_v_dim {head_v_dim}"
        )

    HV = value_dim // head_v_dim
    V = head_v_dim
    fuse_gated_norm = norm_weight is not None
    stride_gate_tok = stride_gate_head = 0
    if fuse_gated_norm:
        if gate is None:
            raise ValueError("gate is required with norm_weight")
        if gate_activation not in ("silu", "sigmoid"):
            raise ValueError(f"unsupported gate_activation {gate_activation!r}")
        if norm_weight.numel() != V or not norm_weight.is_contiguous():
            raise ValueError(
                f"norm_weight must be a contiguous [{V}] tensor, got "
                f"{tuple(norm_weight.shape)}"
            )
        if gate.ndim == 2:
            gate = gate.view(gate.shape[0], HV, V)
        if (
            gate.ndim != 3
            or gate.shape[0] < qkv.shape[0]
            or gate.shape[1:] != (HV, V)
            or gate.stride(2) != 1
        ):
            raise ValueError(
                f"gate must be [T, {HV}, {V}] with unit last stride, got "
                f"{tuple(gate.shape)} strides {gate.stride()}"
            )
        stride_gate_tok, stride_gate_head = gate.stride(0), gate.stride(1)
        if V & (V - 1) or V < 32:
            raise ValueError(
                f"the gated-norm epilogue needs a power-of-2 V >= 32, got {V}"
            )
        T_tok = qkv.shape[0]
        if norm_counter is None:
            norm_counter = torch.zeros(T_tok * HV, dtype=torch.int32, device=qkv.device)
        elif (
            norm_counter.dtype != torch.int32
            or not norm_counter.is_contiguous()
            or norm_counter.numel() < T_tok * HV
        ):
            raise ValueError(
                f"norm_counter must be contiguous int32 with >= {T_tok * HV} "
                f"elements, got {norm_counter.dtype} {tuple(norm_counter.shape)}"
            )
    if (out_fp4 is None) != (out_scale is None):
        raise ValueError("out_fp4 and out_scale go together")
    if out_fp4 is not None:
        if not fuse_gated_norm:
            raise ValueError("out_fp4 needs norm_weight")
        T_tok = qkv.shape[0]
        for name, t, cols in (
            ("out_fp4", out_fp4, HV * V // 2),
            ("out_scale", out_scale, HV * V // 32),
        ):
            if (
                t.dtype != torch.uint8
                or not t.is_contiguous()
                or t.dim() != 2
                or t.shape[0] < T_tok
                or t.shape[1] != cols
            ):
                raise ValueError(
                    f"{name} must be contiguous uint8 [>= {T_tok}, {cols}], got "
                    f"{t.dtype} {tuple(t.shape)}"
                )

    # FlyDSL port (opt-in). Only the speculative-verify shape is routed;
    # everything else falls through to Triton below unchanged.
    if _flydsl_gdr_enabled() and not fuse_gated_norm:
        routed = _try_flydsl_mtp(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            qkv=qkv,
            key_dim=key_dim,
            value_dim=value_dim,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            softplus_beta=beta,
            softplus_threshold=threshold,
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            is_kda=is_kda,
            core_attn_out=core_attn_out,
            draft_window=draft_window,
        )
        if routed is not None:
            return routed

    if scale is None:
        scale = head_k_dim**-0.5
    else:
        assert scale > 0, "scale must be positive"

    B = 1
    T = qkv.shape[0]
    H = key_dim // head_k_dim
    K = head_k_dim
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 4

    if inplace_final_state and ssm_state_indices is None:
        raise ValueError(
            "ssm_state_indices is required when inplace_final_state=True "
            "(kernel indexes final state slots per token)."
        )

    o = (
        core_attn_out.view(-1)[: NK * B * T * HV * V].view(NK, B, T, HV, V)
        if core_attn_out is not None
        else qkv.new_empty(NK, B, T, HV, V)
    )
    if inplace_final_state:
        if initial_state is None:
            raise ValueError("initial_state is required when inplace_final_state=True")
        final_state = initial_state
    else:
        st_dtype = initial_state.dtype if initial_state is not None else qkv.dtype
        final_state = qkv.new_empty(T, HV, V, K, dtype=st_dtype)

    stride_init_state_token = (
        int(initial_state.stride(0)) if initial_state is not None else 0
    )
    stride_final_state_token = int(final_state.stride(0))

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    stride_qkv_l, stride_qkv_hd = qkv.stride()

    grid = (NK, NV, N * HV)
    fused_rearrange_sigmoid_gated_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a.contiguous(),
        b=b.contiguous(),
        dt_bias=dt_bias,
        beta=beta,
        threshold=threshold,
        qkv=qkv,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        norm_weight=norm_weight,
        gate=gate,
        norm_eps=norm_eps,
        norm_counter=norm_counter,
        out_fp4=out_fp4,
        out_scale=out_scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_qkv_l=stride_qkv_l,
        stride_qkv_hd=stride_qkv_hd,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        stride_gate_tok=stride_gate_tok,
        stride_gate_head=stride_gate_head,
        INPLACE_FINAL_STATE=inplace_final_state,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_KDA=is_kda,
        GATE_SIGMOID=gate_activation == "sigmoid",
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, final_state
