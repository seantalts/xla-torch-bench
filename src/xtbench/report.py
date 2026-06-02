"""Format a list of (RowKey, Result) into a human-readable stdout table."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from xtbench.harness import Result


@dataclass(frozen=True)
class RowKey:
    name: str
    shape: tuple
    dtype: str


def _fmt_stat(s) -> str:
    return f"{s.median_ms:.2f} [{s.p10_ms:.2f},{s.p90_ms:.2f}]"


_HEADER = f"{'benchmark':<24} {'shape':<20} {'dtype':<6} {'torch (ms)':<22} {'xla (ms)':<22} {'speedup':<8}"


def format_row(key: RowKey, r: Result) -> str:
    name = key.name[:24]
    shape = str(tuple(key.shape))[:20]
    if not r.equiv_ok:
        return f"{name:<24} {shape:<20} {key.dtype:<6} EQUIV FAIL ({r.equiv_error})"
    speedup = r.torch.median_ms / r.jax.median_ms if r.jax.median_ms > 0 else float("nan")
    return (
        f"{name:<24} {shape:<20} {key.dtype:<6} "
        f"{_fmt_stat(r.torch):<22} {_fmt_stat(r.jax):<22} {speedup:.2f}x"
    )


def format_table(rows: list[tuple[RowKey, Result]]) -> str:
    lines = [_HEADER, "-" * len(_HEADER)]
    for key, r in rows:
        lines.append(format_row(key, r))
    if rows:
        torch_c = median([r.compile_ms_torch for _, r in rows if r.equiv_ok])
        jax_c = median([r.compile_ms_jax for _, r in rows if r.equiv_ok])
        lines.append("")
        lines.append(
            f"compile time (median, first-call): "
            f"torch {torch_c:.0f} ms / xla {jax_c:.0f} ms across {len(rows)} benchmarks"
        )
    return "\n".join(lines)
