"""Apply thread-pinning env vars. MUST be called before importing torch or jax."""
from __future__ import annotations

import os


def pin_threads(n: int) -> None:
    """Pin all CPU thread pools to n threads.

    Must run before torch/jax imports; OMP and MKL respect these vars at
    first allocation, and XLA reads XLA_FLAGS at jaxlib init.
    """
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    # XLA reads --xla_cpu_multi_thread_eigen at jaxlib init; eigen thread count
    # comes from OMP_NUM_THREADS (jaxlib 0.10 dropped --xla_cpu_eigen_num_threads).
    xla_flags = os.environ.get("XLA_FLAGS", "")
    extra = "--xla_cpu_multi_thread_eigen=true"
    os.environ["XLA_FLAGS"] = (xla_flags + " " + extra).strip()


def belt_and_suspenders_torch(n: int) -> None:
    """Call AFTER `import torch` to clamp torch's intra-op pool."""
    import torch
    torch.set_num_threads(n)
