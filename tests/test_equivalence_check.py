import numpy as np
import pytest

from xtbench.equivalence import assert_close, EquivalenceError


def test_close_values_pass():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = a + 1e-6
    assert_close(a, b, atol=1e-4)


def test_far_values_raise():
    a = np.array([1.0], dtype=np.float32)
    b = np.array([2.0], dtype=np.float32)
    with pytest.raises(EquivalenceError, match="max abs diff"):
        assert_close(a, b, atol=1e-4)


def test_shape_mismatch_raises():
    a = np.zeros((2, 2), dtype=np.float32)
    b = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(EquivalenceError, match="shape mismatch"):
        assert_close(a, b, atol=1e-4)
