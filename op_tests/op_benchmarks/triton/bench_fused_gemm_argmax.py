import argparse

import torch
import triton

from aiter.ops.triton.gemm.fused.fused_gemm_argmax import (
    fused_gemm_a16w8_argmax,
    fused_gemm_a16w16_argmax,
)
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype


def _inputs(M, N, K, w_dtype):
    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((N, K), device="cuda", dtype=torch.float32) * 0.02
    if w_dtype == "bf16":
        return x, w.to(torch.bfloat16), None
    fp8 = get_fp8_e4m3_dtype()
    w_scale = w.abs().amax(dim=1) / torch.finfo(fp8).max
    return x, (w / w_scale[:, None]).to(fp8), w_scale


def bench_fn(M, N, K, w_dtype, provider, metric):
    x, w, w_scale = _inputs(M, N, K, w_dtype)
    if provider == "fused":
        if w_scale is None:
            fn = lambda: fused_gemm_a16w16_argmax(x, w)
        else:
            fn = lambda: fused_gemm_a16w8_argmax(x, w, w_scale)
    else:
        w_ref = w if w_scale is None else w.to(torch.bfloat16)
        fn = lambda: torch.nn.functional.linear(x, w_ref).max(dim=-1)
    ms = triton.testing.do_bench_cudagraph(fn)
    if metric == "time":
        return ms
    mem = M * K * x.element_size() + N * K * w.element_size()
    return mem / (ms * 1e-3) * 1e-9


def main():
    parser = argparse.ArgumentParser(description="Fused GEMM and row argmax")
    parser.add_argument("-M", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("-N", type=int, default=124160, help="Vocabulary shard")
    parser.add_argument("-K", type=int, default=2560)
    parser.add_argument("--w-dtype", choices=["bf16", "fp8"], default="bf16")
    parser.add_argument(
        "--metric", choices=["time", "bandwidth"], default="time", help="ms or GB/s"
    )
    args = parser.parse_args()

    benchmark = triton.testing.Benchmark(
        x_names=["M"],
        x_vals=args.M,
        line_arg="provider",
        line_vals=["fused", "linear_max"],
        line_names=["fused", "F.linear + max"],
        ylabel=args.metric,
        plot_name=f"fused_gemm_argmax-{args.w_dtype}-N={args.N}-K={args.K}",
        args={"N": args.N, "K": args.K, "w_dtype": args.w_dtype, "metric": args.metric},
    )
    triton.testing.perf_report([benchmark])(bench_fn).run(print_data=True)


if __name__ == "__main__":
    main()
