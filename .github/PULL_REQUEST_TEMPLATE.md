## Description

<!-- What does this PR do? Why is it needed? -->

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Performance improvement (faster kernels, better scheduling, less memory)
- [ ] Breaking change (fix or feature that breaks existing API)
- [ ] Documentation update
- [ ] Build / CI / tooling

## Checklist

- [ ] `make` compiles with zero warnings
- [ ] `python -m pytest tests/ -v --tb=short` passes
- [ ] New ops include both CPU and CUDA kernels plus a test
- [ ] Bug fixes include a regression test that failed before the fix
- [ ] Code follows project style (2-space C indent, `snake_case`, PEP 8 for Python)

## Performance

<!-- If this is a perf change, include before/after benchmarks. Otherwise delete. -->

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| kernel time | | | |
| memory | | | |

## Related issues

<!-- Link any related issues: Closes #123, Addresses #456 -->
