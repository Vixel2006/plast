import pytest
import numpy as np
import plast

RTOL = 1e-3
ATOL = 1e-4


def has_cuda():
    try:
        import plast

        _ = plast.Device.CUDA
        return True
    except Exception:
        return False


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: marks tests that require CUDA (skipped if no GPU)")


def pytest_collection_modifyitems(config, items):
    if has_cuda():
        return
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)


@pytest.fixture(scope="session", autouse=True)
def arena_setup():
    plast.init_arenas(meta_size_mb=10, data_size_mb=100, device=plast.Device.CPU)
    yield
    plast.reset_transient_arenas()


@pytest.fixture
def device():
    return plast.Device.CPU


@pytest.fixture
def devices():
    if has_cuda():
        return [plast.Device.CPU, plast.Device.CUDA]
    return [plast.Device.CPU]


@pytest.fixture
def tol():
    return {"rtol": RTOL, "atol": ATOL}


@pytest.fixture
def rng():
    return np.random.RandomState(42)
