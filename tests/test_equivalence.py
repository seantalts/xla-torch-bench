"""Rosetta-stone invariant: for every (spec, shape, dtype), torch ≈ jax."""
import numpy as np
import pytest

import benchmarks  # noqa: F401  (populates registry)
from xtbench.equivalence import assert_close
from xtbench.inputs import dtype_atol, to_jax, to_torch
from xtbench.registry import REGISTRY


def _cases():
    for name, spec in REGISTRY:
        for shape in spec.shapes:
            for dtype in spec.dtypes:
                yield pytest.param(name, spec, shape, dtype,
                                   id=f"{name}-{shape}-{dtype}")


@pytest.mark.parametrize("name,spec,shape,dtype", list(_cases()))
def test_equivalence(name, spec, shape, dtype):
    import torch
    import jax

    kwargs = {k: fn(shape) for k, fn in spec.params.items()}
    np_inputs = spec.inputs(shape, dtype, seed=0)

    m = spec.torch_module(**kwargs)
    torch_inputs = [to_torch(a, dtype) for a in np_inputs]
    with torch.no_grad():
        torch_out = m(*torch_inputs).detach().to(torch.float32).numpy()

    f = spec.jax_fn(**kwargs)
    jax_inputs = [to_jax(a, dtype) for a in np_inputs]
    jax_out = np.asarray(f(*jax_inputs), dtype=np.float32)

    assert_close(torch_out, jax_out, atol=dtype_atol(dtype))
