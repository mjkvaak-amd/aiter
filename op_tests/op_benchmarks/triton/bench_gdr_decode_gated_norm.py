# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Decode GDR with and without the gated-RMSNorm epilogue, HIP-graph replay.

Per layer call at decode (one token per sequence), in microseconds:

    gdr            the recurrent kernel alone
    gdr+gnq        the recurrent kernel, then fused_rms_gated_mxfp4_quant
                   (when that op is available), the two launches the MXFP4
                   epilogue replaces
    epi_mxfp4      the recurrent kernel with the MXFP4 epilogue
    epi_bf16       the recurrent kernel with the bf16 gated-norm epilogue

    python bench_gdr_decode_gated_norm.py --n 1 4 8 16 31 64
"""

import argparse

import torch

from aiter.ops.triton.gated_delta_net import fused_rearrange_sigmoid_gated_delta_rule

try:
    from aiter.ops.triton.quant import fused_rms_gated_mxfp4_quant
except ImportError:
    fused_rms_gated_mxfp4_quant = None


def make_inputs(n, h, hv, d, dtype, device="cuda"):
    key_dim, value_dim = h * d, hv * d
    g = torch.Generator(device=device).manual_seed(n)

    def rnd(*shape, scale=1.0, dt=dtype):
        return (torch.randn(*shape, device=device, generator=g) * scale).to(dt)

    return {
        "qkv": rnd(n, 2 * key_dim + value_dim, scale=0.05),
        "A_log": rnd(hv, scale=0.02, dt=torch.float32),
        "a": rnd(n, hv, scale=0.05),
        "b": rnd(n, hv, scale=0.05),
        "dt_bias": rnd(hv, scale=0.005),
        "weight": (1.0 + rnd(d, scale=0.1, dt=torch.float32)).to(dtype),
        "z": rnd(n, value_dim),
        "state": rnd(n + 8, hv, d, d, scale=0.05, dt=torch.float32),
        "slots": torch.arange(n, device=device, dtype=torch.int32),
        "cu_seqlens": torch.arange(n + 1, device=device, dtype=torch.int32),
        "core": torch.empty(n, hv, d, device=device, dtype=dtype),
        "x_q": torch.empty(n, value_dim // 2, dtype=torch.uint8, device=device),
        "x_s": torch.empty(n, value_dim // 32, dtype=torch.uint8, device=device),
        "counter": torch.zeros(n * hv, dtype=torch.int32, device=device),
        "key_dim": key_dim,
        "value_dim": value_dim,
    }


def gdr(inp, d, **norm):
    return fused_rearrange_sigmoid_gated_delta_rule(
        inp["A_log"],
        inp["a"],
        inp["b"],
        inp["dt_bias"],
        inp["qkv"],
        inp["key_dim"],
        inp["value_dim"],
        d,
        d,
        initial_state=inp["state"],
        inplace_final_state=True,
        cu_seqlens=inp["cu_seqlens"],
        ssm_state_indices=inp["slots"],
        use_qk_l2norm_in_kernel=True,
        core_attn_out=inp["core"],
        **norm,
    )


def variants(inp, d, eps, act):
    norm = {
        "norm_weight": inp["weight"],
        "norm_eps": eps,
        "gate": inp["z"],
        "gate_activation": act,
        "norm_counter": inp["counter"],
    }
    out = {"gdr": lambda: gdr(inp, d)}
    if fused_rms_gated_mxfp4_quant is not None:

        def gdr_gnq():
            gdr(inp, d)
            fused_rms_gated_mxfp4_quant(
                inp["core"].view(inp["core"].shape[0], -1),
                inp["weight"],
                inp["z"],
                eps,
                activation=act,
                group_size=d,
            )

        out["gdr+gnq"] = gdr_gnq
    out["epi_mxfp4"] = lambda: gdr(
        inp, d, out_fp4=inp["x_q"], out_scale=inp["x_s"], **norm
    )
    out["epi_bf16"] = lambda: gdr(inp, d, **norm)
    return out


def time_graph(fn, calls=20, replays=50):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (calls * replays)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=[1, 2, 4, 8, 16, 24, 31, 64])
    p.add_argument("--h", type=int, default=8)
    p.add_argument("--hv", type=int, default=24)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--activation", default="sigmoid")
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    names = None
    for n in args.n:
        inp = make_inputs(n, args.h, args.hv, args.d, torch.bfloat16)
        fns = variants(inp, args.d, 1e-6, args.activation)
        if names is None:
            names = list(fns)
            print("| N | " + " | ".join(names) + " |")
            print("| ---: |" + " ---: |" * len(names))
        best = {
            k: min(time_graph(f) for _ in range(args.repeats)) for k, f in fns.items()
        }
        print(f"| {n} | " + " | ".join(f"{best[k]:.2f}" for k in names) + " |")


if __name__ == "__main__":
    main()
