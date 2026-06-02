import pytest
from xtbench.registry import Spec, register, REGISTRY


def setup_function():
    REGISTRY.clear()


def test_register_adds_spec_to_registry():
    @register
    def my_bench():
        return Spec(
            torch_module=lambda d: object(),
            jax_fn=lambda d: (lambda x: x),
            shapes=[(4,)],
            dtypes=["f32"],
            params={"d": lambda s: s[-1]},
        )

    assert len(REGISTRY) == 1
    name, spec = REGISTRY[0]
    assert name == "my_bench"
    assert spec.shapes == [(4,)]
    assert spec.dtypes == ["f32"]


def test_register_rejects_duplicate_names():
    @register
    def dup():
        return Spec(torch_module=None, jax_fn=None, shapes=[(1,)], dtypes=["f32"])

    with pytest.raises(ValueError, match="already registered"):
        @register
        def dup():  # noqa: F811
            return Spec(torch_module=None, jax_fn=None, shapes=[(1,)], dtypes=["f32"])


def test_spec_default_params_and_inputs():
    spec = Spec(
        torch_module=lambda: None, jax_fn=lambda: None,
        shapes=[(8,)], dtypes=["f32"],
    )
    assert spec.params == {}
    assert callable(spec.inputs)
