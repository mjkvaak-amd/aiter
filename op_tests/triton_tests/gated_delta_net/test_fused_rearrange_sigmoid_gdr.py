# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.gated_delta_net import fused_rearrange_sigmoid_gated_delta_rule

cuda_ok = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device required"
)


def _softplus(x: torch.Tensor, beta: float, threshold: float) -> torch.Tensor:
    return torch.where(
        beta * x <= threshold,
        (1.0 / beta) * torch.log1p(torch.exp(beta * x)),
        x,
    )


def ref_fused_rearrange_sigmoid_gdr(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    qkv: torch.Tensor,
    key_dim: int,
    value_dim: int,
    head_k_dim: int,
    head_v_dim: int,
    beta: float,
    threshold: float,
    scale: float,
    initial_state: torch.Tensor | None,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Float reference for decode path (B=1, one sequence), including GQA (HV >= H)."""
    T = qkv.shape[0]
    H = key_dim // head_k_dim
    HV = value_dim // head_v_dim
    K = head_k_dim
    V = head_v_dim
    if HV % H != 0:
        raise ValueError(f"reference expects HV divisible by H, got H={H}, HV={HV}")
    group = HV // H
    B = 1
    o = torch.empty(B, T, HV, V, dtype=torch.float32, device=qkv.device)
    h_state = torch.zeros(HV, V, K, dtype=torch.float32, device=qkv.device)
    if initial_state is not None:
        h_state = initial_state[0].to(torch.float32).clone()

    for t in range(T):
        row = qkv[t]
        for hv in range(HV):
            i_h = hv // group
            q_vec = row[i_h * K : (i_h + 1) * K].float()
            k_vec = row[H * K + i_h * K : H * K + (i_h + 1) * K].float()
            v_vec = row[2 * H * K + hv * V : 2 * H * K + (hv + 1) * V].float()
            b_gate = b[t, hv].float()
            x = a[t, hv].float() + dt_bias[hv].float()
            sp = _softplus(x, beta, threshold)
            g = -torch.exp(A_log[hv].float()) * sp
            beta_out = torch.sigmoid(b_gate)
            if use_qk_l2norm_in_kernel:
                q_vec = q_vec * torch.rsqrt((q_vec * q_vec).sum() + 1e-6)
                k_vec = k_vec * torch.rsqrt((k_vec * k_vec).sum() + 1e-6)
            q_vec = q_vec * scale
            h_sub = h_state[hv]
            h_sub = h_sub * torch.exp(g)
            v_adj = v_vec - (h_sub * k_vec.unsqueeze(0)).sum(dim=-1)
            v_adj = v_adj * beta_out
            h_sub = h_sub + v_adj.unsqueeze(-1) * k_vec.unsqueeze(0)
            out_vec = (h_sub * q_vec.unsqueeze(0)).sum(dim=-1)
            o[0, t, hv] = out_vec
            h_state[hv] = h_sub
    return o, h_state.unsqueeze(0)


# Shapes aligned with ``test_gated_delta_rule.test_fused_recurrent``; dtypes are
# half-precision only — long packed ``T`` with float32 activations tends to blow
# up the recurrent reference / kernel without tighter dynamic-range clamps.
# Each row ends with ``use_qk_l2norm_in_kernel`` (True for stable long-T sweep).
# One small bf16 row uses False to cover the no–L2-norm path (replaces former ``basic``).
_FUSED_GDR_SWEEP = [
    (63, 1, 1, 64, 1, 1, torch.float16, True),
    (500, 4, 4, 60, 1, 1, torch.float16, True),
    (1000, 2, 8, 128, 1, 0.1, torch.float16, True),
    (1024, 2, 2, 128, 0.1, 1, torch.float16, True),
    (1024, 3, 3, 128, 1, 10, torch.float16, True),
    (2048, 4, 4, 64, 0.1, 1, torch.float16, True),
    (1024, 4, 4, 128, 1, 0.1, torch.float16, True),
    (1024, 4, 8, 128, 1, 10, torch.float16, True),
    (1024, 4, 4, 128, 1, 0.1, torch.bfloat16, True),
    (1024, 4, 8, 128, 1, 1, torch.bfloat16, True),
    (2048, 4, 8, 64, 0.1, 1, torch.bfloat16, True),
    (8, 4, 4, 16, 16**-0.5, 1, torch.bfloat16, False),
]


@cuda_ok
@pytest.mark.parametrize(
    (
        "T",
        "H",
        "HV",
        "D",
        "scale",
        "gate_logit_normalizer",
        "dtype",
        "use_qk_l2norm_in_kernel",
    ),
    [
        pytest.param(
            *row,
            id="T{}-H{}-HV{}-D{}-scale{}-gate_logit_normalizer{}-{}-l2{}".format(*row),
        )
        for row in _FUSED_GDR_SWEEP
    ],
)
def test_fused_rearrange_sigmoid_gdr_sweep(
    T: int,
    H: int,
    HV: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
    use_qk_l2norm_in_kernel: bool,
):
    """Shape/dtype sweep aligned with ``test_gated_delta_rule.test_fused_recurrent``."""
    if HV % H != 0:
        pytest.skip("reference/kernel GQA mapping needs HV divisible by H")
    device = "cuda"
    K = V = D
    key_dim = H * K
    value_dim = HV * V

    if use_qk_l2norm_in_kernel:
        torch.manual_seed(42)
        qkv = torch.randn(T, key_dim * 2 + value_dim, device=device, dtype=dtype) * 0.05
        A_log = (
            torch.randn(HV, device=device, dtype=torch.float32).clamp(-2.0, 0.5) * 0.02
        )
        a = (torch.randn(T, HV, device=device, dtype=dtype) * 0.05).clamp(-1.0, 1.0)
        a = a / gate_logit_normalizer
        b_gate = (torch.randn(T, HV, device=device, dtype=dtype) * 0.05).clamp(
            -1.0, 1.0
        )
        dt_bias = (torch.randn(HV, device=device, dtype=dtype) * 0.005).clamp(-0.5, 0.5)
        initial = torch.randn(1, HV, V, K, device=device, dtype=dtype) * 0.05
    else:
        torch.manual_seed(0)
        qkv = torch.randn(T, key_dim * 2 + value_dim, device=device, dtype=dtype)
        A_log = torch.randn(HV, device=device, dtype=torch.float32) * 0.02
        a = torch.randn(T, HV, device=device, dtype=dtype) * 0.1
        a = a / gate_logit_normalizer
        b_gate = torch.randn(T, HV, device=device, dtype=dtype) * 0.1
        dt_bias = torch.randn(HV, device=device, dtype=dtype) * 0.01
        initial = torch.randn(1, HV, V, K, device=device, dtype=dtype)

    o_ref, h_ref = ref_fused_rearrange_sigmoid_gdr(
        A_log,
        a,
        b_gate,
        dt_bias,
        qkv,
        key_dim,
        value_dim,
        K,
        V,
        1.0,
        20.0,
        scale,
        initial,
        use_qk_l2norm_in_kernel,
    )

    core = torch.empty(T, HV, V, device=device, dtype=dtype)
    o_tr, h_tr = fused_rearrange_sigmoid_gated_delta_rule(
        A_log,
        a,
        b_gate,
        dt_bias,
        qkv,
        key_dim,
        value_dim,
        K,
        V,
        beta=1.0,
        threshold=20.0,
        scale=scale,
        initial_state=initial,
        inplace_final_state=False,
        cu_seqlens=None,
        ssm_state_indices=None,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_kda=False,
        core_attn_out=core,
    )

    if dtype == torch.bfloat16:
        rtol, atol = 0.05, 0.1
    elif dtype == torch.float16:
        rtol, atol = 0.03, 0.08
    else:
        rtol, atol = 0.02, 0.05

    if use_qk_l2norm_in_kernel:
        assert torch.isfinite(o_tr.float()).all(), "non-finite Triton output"
        assert torch.isfinite(h_tr.float()).all(), "non-finite Triton final_state"
    torch.testing.assert_close(o_tr.float(), o_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(h_tr[-1].float(), h_ref[0], rtol=rtol, atol=atol)


def _ref_gated_rmsnorm(o, weight, gate, eps, activation):
    o = o.float()
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    g = gate.float()
    return o * (
        torch.sigmoid(g) if activation == "sigmoid" else torch.nn.functional.silu(g)
    )


@cuda_ok
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@pytest.mark.parametrize("gate_2d", [False, True])
@pytest.mark.parametrize(
    "T,H,HV,D",
    [(1, 2, 4, 128), (4, 4, 8, 128), (6, 8, 24, 128), (64, 8, 24, 128), (5, 2, 2, 64)],
)
def test_fused_rearrange_sigmoid_gdr_gated_norm(T, H, HV, D, activation, gate_2d):
    """Decode-style call (one token per sequence, in-place state) with the
    gated RMSNorm epilogue against the float reference + a torch gated norm."""
    device, dtype = "cuda", torch.bfloat16
    K = V = D
    key_dim, value_dim = H * K, HV * V
    eps = 1e-6
    torch.manual_seed(T * 100 + HV)
    qkv = torch.randn(T, key_dim * 2 + value_dim, device=device, dtype=dtype) * 0.05
    A_log = torch.randn(HV, device=device, dtype=torch.float32).clamp(-2.0, 0.5) * 0.02
    a = (torch.randn(T, HV, device=device, dtype=dtype) * 0.05).clamp(-1.0, 1.0)
    b_gate = (torch.randn(T, HV, device=device, dtype=dtype) * 0.05).clamp(-1.0, 1.0)
    dt_bias = (torch.randn(HV, device=device, dtype=dtype) * 0.005).clamp(-0.5, 0.5)
    weight = 1.0 + 0.1 * torch.randn(V, device=device, dtype=dtype)
    # z lives in a wider buffer, as the qkvz split hands it over.
    z_buf = torch.randn(T, HV, V + 16, device=device, dtype=dtype)
    z = z_buf[..., :V]
    if gate_2d:
        z = z.contiguous().view(T, HV * V)
    num_slots = T + 3
    state = torch.randn(num_slots, HV, V, K, device=device, dtype=dtype) * 0.05
    slots = torch.randperm(num_slots, device=device)[:T].to(torch.int32)
    state_ref = state.clone()
    cu_seqlens = torch.arange(T + 1, device=device, dtype=torch.int32)

    o_ref = torch.empty(T, HV, V, device=device, dtype=torch.float32)
    for t in range(T):
        o_t, h_t = ref_fused_rearrange_sigmoid_gdr(
            A_log,
            a[t : t + 1],
            b_gate[t : t + 1],
            dt_bias,
            qkv[t : t + 1],
            key_dim,
            value_dim,
            K,
            V,
            1.0,
            20.0,
            K**-0.5,
            state_ref[slots[t].item()].unsqueeze(0),
            True,
        )
        o_ref[t] = o_t[0, 0]
        state_ref[slots[t].item()] = h_t[0].to(dtype)
    expected = _ref_gated_rmsnorm(o_ref, weight, z.reshape(T, HV, V), eps, activation)

    core = torch.empty(T, HV, V, device=device, dtype=dtype)
    o_tr, _ = fused_rearrange_sigmoid_gated_delta_rule(
        A_log,
        a,
        b_gate,
        dt_bias,
        qkv,
        key_dim,
        value_dim,
        K,
        V,
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=slots,
        use_qk_l2norm_in_kernel=True,
        core_attn_out=core,
        norm_weight=weight,
        norm_eps=eps,
        gate=z,
        gate_activation=activation,
    )
    torch.testing.assert_close(
        o_tr.reshape(T, HV, V).float(), expected, rtol=0.03, atol=0.03
    )
    torch.testing.assert_close(state.float(), state_ref.float(), rtol=0.05, atol=0.02)


def _decode_inputs(T, H, HV, D, seed, device="cuda", dtype=torch.bfloat16):
    K = V = D
    key_dim, value_dim = H * K, HV * V
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape, scale=1.0, dt=dtype):
        return (torch.randn(*shape, device=device, generator=g) * scale).to(dt)

    num_slots = T + 3
    return {
        "qkv": rnd(T, key_dim * 2 + value_dim, scale=0.05),
        "A_log": rnd(HV, scale=0.02, dt=torch.float32).clamp(-2.0, 0.5),
        "a": rnd(T, HV, scale=0.05).clamp(-1.0, 1.0),
        "b": rnd(T, HV, scale=0.05).clamp(-1.0, 1.0),
        "dt_bias": rnd(HV, scale=0.005).clamp(-0.5, 0.5),
        "weight": (1.0 + rnd(V, scale=0.1, dt=torch.float32)).to(dtype),
        "z": rnd(T, HV * V),
        "state": rnd(num_slots, HV, V, K, scale=0.05),
        "slots": torch.randperm(num_slots, device=device, generator=g)[:T].to(
            torch.int32
        ),
        "cu_seqlens": torch.arange(T + 1, device=device, dtype=torch.int32),
        "key_dim": key_dim,
        "value_dim": value_dim,
    }


@cuda_ok
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@pytest.mark.parametrize(
    "T,H,HV,D",
    [
        (1, 8, 24, 128),
        (4, 8, 24, 128),
        (31, 8, 24, 128),
        (64, 8, 24, 128),
        (3, 2, 4, 64),
    ],
)
def test_fused_rearrange_sigmoid_gdr_gated_norm_mxfp4(T, H, HV, D, activation):
    """MXFP4 epilogue: bytes and scales match the separate gated norm + quant
    applied to the kernel's own raw output, the counter is left zeroed, and a
    repeat call on the same counter gives the same result."""
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    device, dtype, eps = "cuda", torch.bfloat16, 1e-6
    V = D
    inp = _decode_inputs(T, H, HV, D, seed=T * 7 + HV)
    counter = torch.zeros(T * HV, dtype=torch.int32, device=device)

    def run(state):
        core = torch.empty(T, HV, V, device=device, dtype=dtype)
        x_q = torch.empty(T, HV * V // 2, dtype=torch.uint8, device=device)
        x_s = torch.empty(T, HV * V // 32, dtype=torch.uint8, device=device)
        o, _ = fused_rearrange_sigmoid_gated_delta_rule(
            inp["A_log"],
            inp["a"],
            inp["b"],
            inp["dt_bias"],
            inp["qkv"],
            inp["key_dim"],
            inp["value_dim"],
            D,
            V,
            initial_state=state,
            inplace_final_state=True,
            cu_seqlens=inp["cu_seqlens"],
            ssm_state_indices=inp["slots"],
            use_qk_l2norm_in_kernel=True,
            core_attn_out=core,
            norm_weight=inp["weight"],
            norm_eps=eps,
            gate=inp["z"],
            gate_activation=activation,
            norm_counter=counter,
            out_fp4=x_q,
            out_scale=x_s,
        )
        return o.reshape(T, HV * V), x_q, x_s

    state0 = inp["state"]
    o1, q1, s1 = run(state0.clone())
    assert int(counter.abs().sum()) == 0
    o2, q2, s2 = run(state0.clone())
    assert int(counter.abs().sum()) == 0
    torch.testing.assert_close(o2, o1, rtol=0, atol=0)
    torch.testing.assert_close(q2, q1, rtol=0, atol=0)
    torch.testing.assert_close(s2, s1, rtol=0, atol=0)

    y = _ref_gated_rmsnorm(
        o1.view(T, HV, V), inp["weight"], inp["z"].view(T, HV, V), eps, activation
    ).to(dtype)
    q_ref = torch.empty_like(q1)
    s_ref = torch.empty_like(s1)
    dynamic_mxfp4_quant(
        y.view(T * HV, V),
        x_fp4=q_ref.view(T * HV, V // 2),
        blockscale_e8m0=s_ref.view(T * HV, V // 32),
    )
    # The fp32 rsqrt differs in the last ulp from torch's, which can move a
    # value across a rounding boundary of the bf16 activation or of E2M1.
    assert (s1 != s_ref).float().mean().item() < 0.01
    assert (q1 != q_ref).float().mean().item() < 0.01
