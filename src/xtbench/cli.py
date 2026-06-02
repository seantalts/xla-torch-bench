"""xtbench CLI. Sets env vars before importing torch/jax."""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Sequence


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="xtbench")
    p.add_argument("--filter", default=".*",
                   help="regex on benchmark name (default: match all)")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1,
                   help="thread pool size for torch/jax/OpenMP")
    p.add_argument("--dtypes", default="",
                   help="comma list of dtypes (default: all in spec)")
    p.add_argument("--shapes", default="",
                   help="comma list of '(d0,d1,...)' tuples; overrides spec shapes")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=50)
    return p.parse_args(argv)


def _parse_shapes(s: str) -> list[tuple] | None:
    if not s.strip():
        return None
    out = []
    for chunk in s.split("),"):
        chunk = chunk.strip().strip("()")
        if not chunk:
            continue
        out.append(tuple(int(x) for x in chunk.split(",") if x.strip()))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    # Pin threads BEFORE importing torch/jax.
    from xtbench.env import pin_threads, belt_and_suspenders_torch
    pin_threads(args.threads)

    # Now safe to import everything else.
    import benchmarks  # noqa: F401  (populates registry as a side effect)
    from xtbench.harness import run_benchmark
    from xtbench.registry import REGISTRY
    from xtbench.report import RowKey, format_table

    belt_and_suspenders_torch(args.threads)

    pat = re.compile(args.filter)
    dtype_filter = set(d.strip() for d in args.dtypes.split(",") if d.strip())
    shape_override = _parse_shapes(args.shapes)

    rows = []
    for name, spec in REGISTRY:
        if not pat.search(name):
            continue
        shapes = shape_override if shape_override else spec.shapes
        for shape in shapes:
            for dtype in spec.dtypes:
                if dtype_filter and dtype not in dtype_filter:
                    continue
                result = run_benchmark(
                    spec, shape=shape, dtype=dtype,
                    warmup=args.warmup, iters=args.iters,
                )
                rows.append((RowKey(name, shape, dtype), result))

    print(format_table(rows))
    return 0
