import argparse

import torch
import triton

from aiter.ops.triton.gemm.fused.fused_gemm_mtp_residual_linear_shared import (
    fused_gemm_mtp_residual_linear_shared,
)


def _gemma_rmsnorm(x, weight, eps):
    x32 = x.float()
    y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (y * (1 + weight.float())).to(x.dtype)


def bench_fn(T, C, N, K, provider, metric):
    eps = 1e-6
    e = torch.randn((T, K), device="cuda", dtype=torch.bfloat16)
    h = torch.randn((T, C * K), device="cuda", dtype=torch.bfloat16)
    e_nw = torch.randn((K,), device="cuda", dtype=torch.bfloat16) * 0.1
    h_nw = torch.randn((C * K,), device="cuda", dtype=torch.bfloat16) * 0.1
    w_e = torch.randn((N, K), device="cuda", dtype=torch.bfloat16) * K**-0.5
    w_h = torch.randn((N, K), device="cuda", dtype=torch.bfloat16) * K**-0.5
    if provider == "fused":
        fn = lambda: fused_gemm_mtp_residual_linear_shared(
            e, h, e_nw, h_nw, w_e, w_h, eps
        )
    else:

        def fn():
            pe = torch.nn.functional.linear(_gemma_rmsnorm(e, e_nw, eps), w_e)
            yh = _gemma_rmsnorm(h, h_nw, eps).view(T, C, K)
            return (pe[:, None] + torch.nn.functional.linear(yh, w_h)).flatten(1)

    ms = triton.testing.do_bench_cudagraph(fn)
    if metric == "time":
        return ms
    mem = 2 * N * K * 2 + (T * K + T * C * K + T * C * N) * 2
    return mem / (ms * 1e-3) * 1e-9


def main():
    parser = argparse.ArgumentParser(description="Fused MTP input projection")
    parser.add_argument("-T", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("-C", type=int, default=4, help="Hyper-connection streams")
    parser.add_argument("-N", type=int, default=2560)
    parser.add_argument("-K", type=int, default=2560)
    parser.add_argument("--metric", choices=["time", "bandwidth"], default="time")
    args = parser.parse_args()
    benchmark = triton.testing.Benchmark(
        x_names=["T"],
        x_vals=args.T,
        line_arg="provider",
        line_vals=["fused", "torch"],
        line_names=["fused", "norms + F.linear + add"],
        ylabel=args.metric,
        plot_name=f"fused_gemm_mtp_residual_linear_shared-C={args.C}-N={args.N}-K={args.K}",
        args={"C": args.C, "N": args.N, "K": args.K, "metric": args.metric},
    )
    triton.testing.perf_report([benchmark])(bench_fn).run(print_data=True)


if __name__ == "__main__":
    main()
