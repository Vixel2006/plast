# contributing

thanks for wanting to make plast faster.

this is a systems-level project. every abstraction has a cost. every kernel launch has overhead. write code like you're paying for the silicon.

## philosophy

before you open a pr, ask yourself:

- **does this belong in the hot path?** if yes, it should be in C/CUDA, not python.
- **is this fusable?** if two ops touch the same memory, they should be one kernel.
- **does this allocate?** it shouldn't. use the arena.

the classic api (`nn.Module` etc) exists for accessibility. the real work is in the scheduler, the JIT compiler, and the kernels. that's where performance lives.

## getting started

```sh
git clone https://github.com/Vixel2006/plast.git
cd plast
make
python test_xor.py
```

## dev workflow

- branch from `main`: `git checkout -b feat/your-thing`
- one feature per pr. one bug per pr. keep it focused.
- `make` must compile cleanly with zero warnings.
- `python -m pytest tests/ -v --tb=short` must pass before you open the pr.
- new ops ship with **both** cpu and cuda kernels, plus a test.

## code style

| language | convention |
|----------|-----------|
| **c** | 2-space indent, `snake_case`, braces on same line, short functions |
| **cuda** | same as c. shared memory + coalesced access or it doesn't land |
| **python** | pep 8. type annotations. zero python in the hot path. |

the test suite is your style guide. match what's there.

## what we look for

- **performance first** — this is not a toy framework. if your change makes things slower, explain why the tradeoff is worth it.
- **kernels** — cpu + cuda, forward + backward, contiguous fast path + broadcast fallback.
- **fusions** — prefer adding fusion opportunities to the JIT over writing more separate ops.
- **no new dependencies** — plast is zero-dependency in C. keep it that way.
- **bug fixes** — must include a regression test that failed before your fix.

## opening a PR

use the [PR template](.github/PULL_REQUEST_TEMPLATE.md) — it covers the checklist automatically.

- branch from `main`: `git checkout -b feat/your-thing`
- one change per PR. focused reviews land faster.
- `make` must compile with zero warnings.
- `python -m pytest tests/ -v --tb=short` must pass before opening.
- bug fixes include a regression test that failed before the fix.
- perf changes include before/after numbers in the template.

## reporting issues

use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.yaml). include:

- a minimal reproduction (10 lines or fewer)
- what you expected
- what actually happened
- `nvcc --version` and `gcc --version`

features and ideas use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.yaml).

## the pipeline philosophy

if you're adding a feature, ask: does this belong in the oop path or the pipeline path?

- **oop path** (`plast.nn.Module` etc): for users who want pytorch-like ergonomics. maintain it, but don't optimize it heavily — it's a convenience layer.
- **pipeline path** (`plast.pipeline`, scheduler, JIT): this is where plast differentiates. new capabilities should live here first, then be exposed through the oop api if it makes sense.

the pipeline is lazy. it composes functions, builds a DAG, and JIT-compiles fused kernels. adding a new op? make sure the scheduler can see it and the JIT can fuse it.

## license

by contributing, you agree your contributions are MIT-licensed. same as the rest of the project.

now go write some fast code.
