# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.gemm.fused.fused_gemm_mtp_residual_linear_shared import (
    fused_gemm_mtp_residual_linear_shared,
)


def _gemma_rmsnorm(x, weight, eps):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1 + weight.float())


def _ref(e, h, e_nw, h_nw, w_e, w_h, eps):
    T, K = e.shape
    C = h.shape[1] // K
    proj_e = _gemma_rmsnorm(e, e_nw, eps) @ w_e.float().T
    proj_h = _gemma_rmsnorm(h, h_nw, eps).view(T, C, K) @ w_h.float().T
    return (proj_h + proj_e[:, None, :]).reshape(T, -1)


@pytest.mark.parametrize("T", [1, 3, 4, 16, 33, 64])
@pytest.mark.parametrize("C, N, K", [(4, 2560, 2560), (4, 512, 384), (2, 256, 200)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_gemm_mtp_residual_linear_shared(T, C, N, K, dtype):
    torch.manual_seed(0)
    eps = 1e-6
    e = torch.randn((T, K), device="cuda", dtype=dtype)
    h = torch.randn((T, C * K), device="cuda", dtype=dtype) * 3
    e_nw = torch.randn((K,), device="cuda", dtype=dtype) * 0.1
    h_nw = torch.randn((C * K,), device="cuda", dtype=dtype) * 0.1
    w_e = torch.randn((N, K), device="cuda", dtype=dtype) * K**-0.5
    w_h = torch.randn((N, K), device="cuda", dtype=dtype) * K**-0.5
    out = fused_gemm_mtp_residual_linear_shared(e, h, e_nw, h_nw, w_e, w_h, eps)
    ref = _ref(e, h, e_nw, h_nw, w_e, w_h, eps)
    assert out.shape == (T, C * N) and out.dtype == dtype
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=2e-2)
