# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.gemm.fused.fused_gemm_argmax import (
    argmax_pair_pick,
    fused_gemm_a16w8_argmax,
    fused_gemm_a16w16_argmax,
)
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype


def _ref_logits(x, w):
    return (x.float() @ w.float().T).to(x.dtype).float()


def _check_argmax(got, logits, index_offset):
    ref_val, _ = logits.max(dim=-1)
    idx = got[:, 1].long() - index_offset
    torch.testing.assert_close(got[:, 0], ref_val, atol=0, rtol=2**-7)
    # The selected logit is the maximum, up to the rounding of one logit.
    picked = logits.gather(1, idx[:, None]).squeeze(1)
    torch.testing.assert_close(picked, ref_val, atol=0, rtol=2**-7)


@pytest.mark.parametrize("M", [1, 4, 12, 16, 33, 64])
@pytest.mark.parametrize("N, K", [(4096, 512), (124160, 2560), (1000, 300)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_gemm_a16w16_argmax(M, N, K, dtype):
    torch.manual_seed(0)
    x = torch.randn((M, K), device="cuda", dtype=dtype)
    w = torch.randn((N, K), device="cuda", dtype=dtype) * 0.05
    got = fused_gemm_a16w16_argmax(x, w, index_offset=N)
    _check_argmax(got, _ref_logits(x, w), N)


@pytest.mark.parametrize("M", [1, 4, 16, 64])
@pytest.mark.parametrize("N, K", [(4096, 512), (124160, 2560)])
@pytest.mark.parametrize("transposed", [False, True])
def test_fused_gemm_a16w8_argmax(M, N, K, transposed):
    torch.manual_seed(0)
    fp8 = get_fp8_e4m3_dtype()
    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((N, K), device="cuda", dtype=torch.float32) * 0.05
    w_scale = w.abs().amax(dim=1) / torch.finfo(fp8).max
    w_q = (w / w_scale[:, None]).to(fp8)
    if transposed:
        w_q = w_q.T.contiguous().T
    got = fused_gemm_a16w8_argmax(x, w_q, w_scale, index_offset=7)
    w_deq = w_q.float() * w_scale[:, None]
    logits = (x.float() @ w_deq.T).to(x.dtype).float()
    _check_argmax(got, logits, 7)


def test_fused_gemm_a16w16_argmax_ties_pick_first():
    torch.manual_seed(0)
    M, N, K = 8, 8192, 256
    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((N, K), device="cuda", dtype=torch.bfloat16) * 0.05
    top = torch.randn((1, K), device="cuda", dtype=torch.bfloat16)
    x[:] = top
    for n in (77, 3000, 8100):
        w[n] = top[0]
    got = fused_gemm_a16w16_argmax(x, w)
    assert torch.equal(got[:, 1].long(), torch.full((M,), 77, device="cuda"))


def test_fused_gemm_a16w16_argmax_ignores_padding_rows():
    torch.manual_seed(0)
    M, N, K, n_valid = 4, 4096, 256, 4000
    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((N, K), device="cuda", dtype=torch.bfloat16) * 0.05
    w[n_valid:] = x[0] * 10
    got = fused_gemm_a16w16_argmax(x, w, n_valid=n_valid)
    assert (got[:, 1] < n_valid).all()
    _check_argmax(got, _ref_logits(x, w[:n_valid]), 0)


@pytest.mark.parametrize("M", [1, 16, 64])
@pytest.mark.parametrize("T", [1, 2, 3, 8])
def test_argmax_pair_pick(M, T):
    torch.manual_seed(0)
    pairs = torch.randn((M, T, 2), device="cuda")
    pairs[..., 1] = torch.randint(0, 2**24, (M, T), device="cuda").float()
    pairs[0, :, 0] = 1.0
    rank = pairs[..., 0].argmax(dim=-1, keepdim=True)
    want = pairs[..., 1].gather(1, rank).squeeze(1).long()
    got = argmax_pair_pick(pairs.view(M, 2 * T))
    assert torch.equal(got, want)
