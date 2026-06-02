import time
import numpy as np
import pytest

from xtbench.timing import time_torch, time_jax, Stats, summarize


def test_summarize_basic():
    s = summarize([1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000])
    assert isinstance(s, Stats)
    assert s.median_ms == pytest.approx(3.0, rel=1e-6)
    assert s.p10_ms == pytest.approx(1.0, rel=1e-6)
    assert s.p90_ms == pytest.approx(5.0, rel=1e-6)


def test_time_torch_runs_callable():
    import torch
    x = torch.zeros(4)
    elapsed = time_torch(lambda: x + 1)
    assert elapsed > 0


def test_time_jax_runs_callable_and_blocks():
    import jax.numpy as jnp
    x = jnp.zeros(4)
    elapsed = time_jax(lambda: x + 1)
    assert elapsed > 0
