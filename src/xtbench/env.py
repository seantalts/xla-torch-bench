"""Apply thread-pinning env vars. MUST be called before importing torch or jax."""
from __future__ import annotations

import os


def pin_threads(n: int) -> None:
    """Pin all CPU thread pools to n threads.

    Must run before torch/jax imports; OMP and MKL respect these vars at
    first allocation. PJRT_NPROC / NPROC are checked by XLA's
    `DefaultThreadPoolSize` in xla/pjrt/utils.cc to size the PJRT CPU client's
    `eigen_intraop_pool_` — without them XLA uses the full host CPU count and
    timings are not comparable to OMP-pinned PyTorch.
    """
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["PJRT_NPROC"] = str(n)
    os.environ["NPROC"] = str(n)
    xla_flags = os.environ.get("XLA_FLAGS", "")
    extra = "--xla_cpu_multi_thread_eigen=true"
    os.environ["XLA_FLAGS"] = (xla_flags + " " + extra).strip()


def belt_and_suspenders_torch(n: int) -> None:
    """Call AFTER `import torch` to clamp torch's intra-op pool."""
    import torch
    torch.set_num_threads(n)
