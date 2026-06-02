from xtbench.harness import Result
from xtbench.report import format_row, format_table, RowKey
from xtbench.timing import Stats


def _result(t_med, j_med):
    return Result(
        torch=Stats(median_ms=t_med, p10_ms=t_med * 0.95, p90_ms=t_med * 1.05),
        jax=Stats(median_ms=j_med, p10_ms=j_med * 0.95, p90_ms=j_med * 1.05),
        compile_ms_torch=100.0, compile_ms_jax=80.0,
        equiv_ok=True,
    )


def test_format_row_speedup_gt_1():
    key = RowKey(name="rms_norm", shape=(1, 1024, 768), dtype="f32")
    row = format_row(key, _result(t_med=1.0, j_med=0.5))
    assert "rms_norm" in row
    assert "2.00x" in row


def test_format_row_equiv_fail():
    key = RowKey(name="bad", shape=(8,), dtype="f32")
    r = Result(
        torch=Stats(0, 0, 0), jax=Stats(0, 0, 0),
        compile_ms_torch=0, compile_ms_jax=0,
        equiv_ok=False, equiv_error="max abs diff 1.0",
    )
    row = format_row(key, r)
    assert "EQUIV FAIL" in row


def test_format_table_has_header_and_compile_summary():
    rows = [
        (RowKey("rms_norm", (1, 1024, 768), "f32"), _result(1.0, 0.5)),
        (RowKey("rms_norm", (1, 1024, 768), "bf16"), _result(0.8, 0.4)),
    ]
    out = format_table(rows)
    assert "benchmark" in out and "speedup" in out
    assert "rms_norm" in out
    assert "compile time" in out
