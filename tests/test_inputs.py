import numpy as np
import pytest

from xtbench.inputs import to_torch, to_jax, dtype_atol


def test_to_torch_f32_roundtrip():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    t = to_torch(arr, "f32")
    assert tuple(t.shape) == (2, 2)
    assert str(t.dtype) == "torch.float32"
    np.testing.assert_array_equal(t.numpy(), arr)


def test_to_torch_bf16_downcasts():
    arr = np.array([1.0, 2.0], dtype=np.float32)
    t = to_torch(arr, "bf16")
    assert str(t.dtype) == "torch.bfloat16"


def test_to_jax_f32_roundtrip():
    arr = np.array([1.0, 2.0], dtype=np.float32)
    j = to_jax(arr, "f32")
    assert str(j.dtype) == "float32"
    np.testing.assert_array_equal(np.asarray(j), arr)


def test_to_jax_bf16_downcasts():
    arr = np.array([1.0, 2.0], dtype=np.float32)
    j = to_jax(arr, "bf16")
    assert str(j.dtype) == "bfloat16"


def test_dtype_atol():
    assert dtype_atol("f32") == 1e-4
    assert dtype_atol("bf16") == 5e-2


def test_unknown_dtype_raises():
    with pytest.raises(ValueError, match="unknown dtype"):
        to_torch(np.zeros(1, dtype=np.float32), "f16")
