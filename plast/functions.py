from typing import Any
import time # Added for benchmarking

import numpy as np

import plast._plast_cpp_core as core_bindings
from plast.core.device import Device
from plast.core.tensor import Tensor
from plast.ops import add as ops_add
from plast.ops import sub as ops_sub
from plast.ops import matmul as ops_matmul # Added for matmul operation

# =========== Binary Operations =============
def add(a: Tensor | float, b: Tensor | float) -> Tensor:
    return ops_add(a, b)


def sub(a: Tensor | float, b: Tensor | float) -> Tensor:
    if isinstance(a, Tensor):
        return ops_sub(a, b)

def matmul(a: Tensor, b: Tensor) -> Tensor: # Added matmul function
    return ops_matmul(a, b)


# =========== Initialization Operations ============
# Map Python types to C++ DTypes
_DTYPE_MAP = {
    np.float32: core_bindings.DType.FLOAT32,
    np.float64: core_bindings.DType.FLOAT64,
    np.int8: core_bindings.DType.INT8,
    np.int16: core_bindings.DType.INT16,
    np.int32: core_bindings.DType.INT32,
    np.int64: core_bindings.DType.INT64,
    np.uint8: core_bindings.DType.UINT8,
    np.uint16: core_bindings.DType.UINT16,
    np.uint32: core_bindings.DType.UINT32,
    np.uint64: core_bindings.DType.UINT64,
    np.bool_: core_bindings.DType.BOOL,
}

# Map Python device strings to C++ DeviceType
_DEVICE_MAP = {
    "cpu": core_bindings.DeviceType.CPU,
    "cuda": core_bindings.DeviceType.CUDA,
}


def zeros(
    shape: tuple[int, ...] | list[int],
    dtype: Any = np.float32,
    device: str | Device = "cpu",
) -> Tensor:
    if isinstance(device, str):
        device = Device.parse(device)

    cpp_dtype = _DTYPE_MAP.get(dtype)
    if cpp_dtype is None:
        raise ValueError(f"Unsupported numpy dtype: {dtype}")

    cpp_device = _DEVICE_MAP.get(device.type)
    if cpp_device is None:
        raise ValueError(f"Unsupported device: {device.type}")

    cpp_tensor = core_bindings.zeros(list(shape), cpp_dtype, cpp_device)
    return Tensor(cpp_node=core_bindings.Node(cpp_tensor))


def ones(
    shape: tuple[int, ...] | list[int],
    dtype: Any = np.float32,
    device: str | Device = "cpu",
) -> Tensor:
    if isinstance(device, str):
        device = Device.parse(device)

    cpp_dtype = _DTYPE_MAP.get(dtype)
    if cpp_dtype is None:
        raise ValueError(f"Unsupported numpy dtype: {dtype}")

    cpp_device = _DEVICE_MAP.get(device.type)
    if cpp_device is None:
        raise ValueError(f"Unsupported device: {device.type}")

    cpp_tensor = core_bindings.ones(list(shape), cpp_dtype, cpp_device)
    return Tensor(cpp_node=core_bindings.Node(cpp_tensor))


def randn(
    shape: tuple[int, ...] | list[int],
    seed: int = 42,
    dtype: Any = np.float32,
    device: str | Device = "cpu",
) -> Tensor:
    if isinstance(device, str):
        device = Device.parse(device)

    cpp_dtype = _DTYPE_MAP.get(dtype)
    if cpp_dtype is None:
        raise ValueError(f"Unsupported numpy dtype: {dtype}")

    cpp_device = _DEVICE_MAP.get(device.type)
    if cpp_device is None:
        raise ValueError(f"Unsupported device: {device.type}")

    cpp_tensor = core_bindings.randn(list(shape), cpp_dtype, cpp_device, seed)
    return Tensor(cpp_node=core_bindings.Node(cpp_tensor))


def uniform(
    shape: tuple[int, ...] | list[int],
    low: float = 0.0,
    high: float = 1.0,
    dtype: Any = np.float32,
    device: str | Device = "cpu",
    seed: int = 42, # Added seed argument
) -> Tensor:
    if isinstance(device, str):
        device = Device.parse(device)

    cpp_dtype = _DTYPE_MAP.get(dtype)
    if cpp_dtype is None:
        raise ValueError(f"Unsupported numpy dtype: {dtype}")

    cpp_device = _DEVICE_MAP.get(device.type)
    if cpp_device is None:
        raise ValueError(f"Unsupported device: {device.type}")

    cpp_tensor = core_bindings.uniform(list(shape), cpp_dtype, cpp_device, low, high, seed) # Pass seed
    return Tensor(cpp_node=core_bindings.Node(cpp_tensor))


def from_data(
    shape: tuple[int, ...] | list[int],
    data: list[int] | list[float] | np.ndarray,
    dtype: Any = np.float32,
    device: str | Device = "cpu",
) -> Tensor:
    if isinstance(device, str):
        device = Device.parse(device)

    if isinstance(data, (list, tuple)):
        data = np.array(data, dtype=dtype)
    elif not isinstance(data, np.ndarray):
        raise TypeError(
            f"Unsupported data type for from_data: {type(data)}. Expected list, tuple, or numpy.ndarray."
        )

    if dtype is None:
        dtype = data.dtype
    else:
        data = data.astype(dtype)

    cpp_dtype = _DTYPE_MAP.get(dtype)
    if cpp_dtype is None:
        raise ValueError(f"Unsupported numpy dtype: {dtype}")

    cpp_device = _DEVICE_MAP.get(device.type)
    if cpp_device is None:
        raise ValueError(f"Unsupported device: {device.type}")

    cpp_tensor = core_bindings.from_data(data, list(shape), cpp_dtype, cpp_device)
    return Tensor(cpp_node=core_bindings.Node(cpp_tensor))


def benchmark_matmul():
    shape = (1024, 1024)

    print(f"Benchmarking Matmul for shape {shape}...")

    # CPU Benchmark
    start_time = time.time()
    a_cpu = uniform(shape, device="cpu")
    b_cpu = uniform(shape, device="cpu")
    result_cpu = matmul(a_cpu, b_cpu)
    # Ensure computation is finished if it's lazy
    _ = result_cpu.data
    end_time = time.time()
    print(f"CPU Matmul took: {end_time - start_time:.4f} seconds")

    # CUDA Benchmark
    try:
        start_time = time.time()
        a_cuda = uniform(shape, device="cuda")
        b_cuda = uniform(shape, device="cuda")
        result_cuda = matmul(a_cuda, b_cuda)
        # Ensure computation is finished if it's lazy
        _ = result_cuda.data
        end_time = time.time()
        print(f"CUDA Matmul took: {end_time - start_time:.4f} seconds")
    except ValueError as e:
        print(f"CUDA benchmark skipped: {e}")


if __name__ == "__main__":
    benchmark_matmul()
