# xla-torch-bench

CPU benchmarks comparing XLA:CPU against PyTorch's compiled (Inductor) backend on a shared set of "rosetta-stone" workloads — each benchmark is written twice, once as a PyTorch `nn.Module` and once as a JAX function, expressing the same compute idiomatically in each framework.

Status: design phase. See [docs/superpowers/specs/2026-06-01-xla-pytorch-cpu-benchmarks-design.md](docs/superpowers/specs/2026-06-01-xla-pytorch-cpu-benchmarks-design.md).
