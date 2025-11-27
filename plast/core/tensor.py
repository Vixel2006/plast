from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

# Import the new pybind11 module
from .. import _plast_cpp_core

# Map Python types to C++ DTypes
_DTYPE_MAP = {
    np.float32: _plast_cpp_core.DType.FLOAT32,
    np.float64: _plast_cpp_core.DType.FLOAT64,
    np.int8: _plast_cpp_core.DType.INT8,
    np.int16: _plast_cpp_core.DType.INT16,
    np.int32: _plast_cpp_core.DType.INT32,
    np.int64: _plast_cpp_core.DType.INT64,
    np.uint8: _plast_cpp_core.DType.UINT8,
    np.uint16: _plast_cpp_core.DType.UINT16,
    np.uint32: _plast_cpp_core.DType.UINT32,
    np.uint64: _plast_cpp_core.DType.UINT64,
    np.bool_: _plast_cpp_core.DType.BOOL,
    # Add more as needed
}

# Reverse map for converting C++ DType to numpy dtype
_REVERSE_DTYPE_MAP = {v: k for k, v in _DTYPE_MAP.items()}


# Global execution engine instance
_execution_engine = _plast_cpp_core.ExecutionEngine()


class Tensor:
    _cpp_node: Any  # This will hold std::shared_ptr<plast::graph::Node>

    def __init__(
        self,
        data: Optional[Any] = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[Any] = None,
        device: Optional[str] = None,
        cpp_node: Optional[Any] = None,
    ):
        """
        Initializes a Tensor.
        If cpp_node is provided, it wraps an existing C++ Node.
        If data is provided, it creates a new leaf Node from data.
        If neither cpp_node nor data is provided, it creates an uninitialized C++ Tensor
        with the given shape, dtype, and device.
        """
        if cpp_node is not None:
            self._cpp_node = cpp_node
        elif data is not None:
            # Existing logic for data-based initialization
            if isinstance(data, (list, tuple)):
                data = np.array(data, dtype=dtype)
            elif not isinstance(data, np.ndarray):
                raise TypeError(f"Unsupported data type: {type(data)}")

            if dtype is None:
                dtype = data.dtype
            else:
                data = data.astype(dtype)

            if shape is None:
                shape = data.shape
            else:
                if data.shape != shape:
                    raise ValueError(
                        f"Provided data shape {data.shape} does not match specified shape {shape}"
                    )

            # Ensure dtype is a numpy.dtype object before accessing .type
            if not isinstance(dtype, np.dtype):
                dtype = np.dtype(dtype)

            cpp_dtype = _DTYPE_MAP.get(dtype.type)
            if cpp_dtype is None:
                raise ValueError(f"Unsupported numpy dtype: {dtype}")

            cpp_device = _plast_cpp_core.DeviceType.CPU  # Default to CPU for now
            if device == "cuda":
                cpp_device = _plast_cpp_core.DeviceType.CUDA
            elif device is not None and device != "cpu":
                raise ValueError(f"Unsupported device: {device}")

            # Create a C++ Tensor object from numpy data
            # This part needs careful implementation in C++ to avoid double-free or memory leaks.
            # For now, we'll create a C++ Tensor that allocates memory, and we'll
            # assume the data is copied into it later.
            cpp_tensor_value = _plast_cpp_core.Tensor(
                list(shape), cpp_dtype, cpp_device
            )
            self._cpp_node = _plast_cpp_core.Node(cpp_tensor_value)
        else:
            # New logic for shape-based initialization (uninitialized tensor)
            if shape is None:
                raise ValueError(
                    "Either 'data', 'cpp_node', or 'shape' must be provided for Tensor initialization."
                )
            if dtype is None:
                # Default dtype if not provided for uninitialized tensor
                dtype = np.float32

            # Ensure dtype is a numpy.dtype object before accessing .type
            if not isinstance(dtype, np.dtype):
                dtype = np.dtype(dtype)

            cpp_dtype = _DTYPE_MAP.get(dtype.type)
            if cpp_dtype is None:
                raise ValueError(f"Unsupported numpy dtype: {dtype}")

            cpp_device = _plast_cpp_core.DeviceType.CPU
            if device == "cuda":
                cpp_device = _plast_cpp_core.DeviceType.CUDA
            elif device != "cpu":
                raise ValueError(f"Unsupported device: {device}")

            # Create a C++ Tensor object that allocates memory based on shape, dtype, device
            cpp_tensor_value = _plast_cpp_core.Tensor(
                list(shape), cpp_dtype, cpp_device
            )
            self._cpp_node = _plast_cpp_core.Node(cpp_tensor_value)

    @property
    def data(self) -> np.ndarray:
        """
        Triggers computation of the graph up to this node and returns the data as a numpy array.
        """
        cpp_tensor = _execution_engine.execute(self._cpp_node)

        # Placeholder: In a real scenario, you'd copy data from C++ Tensor to numpy array
        # or create a view if possible.
        return np.zeros(
            shape=tuple(cpp_tensor.shape),
            dtype=_REVERSE_DTYPE_MAP.get(cpp_tensor.dtype),
        )  # Placeholder

    @property
    def shape(self) -> Tuple[int, ...]:
        # Shape is known even before execution for most ops
        # This would require the C++ Node to store/infer its output shape
        # For now, we'll execute to get the shape from the resulting tensor
        cpp_tensor = _execution_engine.execute(self._cpp_node)
        return tuple(cpp_tensor.shape)

    @property
    def dtype(self) -> Any:
        cpp_tensor = _execution_engine.execute(self._cpp_node)
        return _REVERSE_DTYPE_MAP.get(cpp_tensor.dtype)

    @property
    def device(self) -> str:
        cpp_tensor = _execution_engine.execute(self._cpp_node)
        if cpp_tensor.device == _plast_cpp_core.DeviceType.CPU:
            return "cpu"
        elif cpp_tensor.device == _plast_cpp_core.DeviceType.CUDA:
            return "cuda"
        return "unknown"

    def to(self, device: str) -> Tensor:
        # This would create a new C++ Node for device transfer
        # For now, return self
        return self

    def __add__(self, other: Tensor | float) -> Tensor:
        if isinstance(other, Tensor):
            new_cpp_node = _plast_cpp_core.add_op_node(self._cpp_node, other._cpp_node)
            return Tensor(cpp_node=new_cpp_node)
        elif isinstance(other, (int, float)):
            # Handle scalar addition by creating a scalar tensor
            scalar_tensor = Tensor(
                data=np.array(other, dtype=np.float32)
            )  # Assuming float32 for scalar
            new_cpp_node = _plast_cpp_core.add_op_node(
                self._cpp_node, scalar_tensor._cpp_node
            )
            return Tensor(cpp_node=new_cpp_node)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for +: 'Tensor' and '{type(other)}'"
            )

    def __sub__(self, other: Tensor | float) -> Tensor:
        if isinstance(other, Tensor):
            new_cpp_node = _plast_cpp_core.sub_op_node(self._cpp_node, other._cpp_node)
            return Tensor(cpp_node=new_cpp_node)
        elif isinstance(other, (int, float)):
            # Handle scalar subtraction by creating a scalar tensor
            scalar_tensor = Tensor(
                data=np.array(other, dtype=np.float32)
            )  # Assuming float32 for scalar
            new_cpp_node = _plast_cpp_core.sub_op_node(
                self._cpp_node, scalar_tensor._cpp_node
            )
            return Tensor(cpp_node=new_cpp_node)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for -: 'Tensor' and '{type(other)}'"
            )

    def __repr__(self) -> str:
        # For repr, we might not want to trigger full execution.
        # This would require the C++ Node to expose its inferred shape/dtype without execution.
        # For now, we'll execute to get details.
        try:
            cpp_tensor = _execution_engine.execute(self._cpp_node)
            return f"Tensor(shape={tuple(cpp_tensor.shape)}, dtype={_REVERSE_DTYPE_MAP.get(cpp_tensor.dtype)}, device={self.device})"
        except Exception as e:
            return f"Tensor(uncomputed_node, error_on_repr: {e})"

    # Placeholder for other methods
    def numel(self) -> int:
        cpp_tensor = _execution_engine.execute(self._cpp_node)
        return cpp_tensor.num_elements()

    def __del__(self):
        # C++ shared_ptr handles memory, so no explicit free needed here.
        pass


# Example usage (for testing the Python side)
if __name__ == "__main__":
    # Create some tensors
    a = Tensor(data=[[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = Tensor(data=[[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    # Perform an operation
    c = a + b

    # Access data (triggers execution)
    print("Result of a + b:")
    print(c.data)
    print(f"Shape: {c.shape}, DType: {c.dtype}, Device: {c.device}")

    # Add with scalar
    d = c + 10.0
    print("\nResult of c + 10.0:")
    print(d.data)
    print(f"Shape: {d.shape}, DType: {d.dtype}, Device: {d.device}")

    # Test device transfer (placeholder)
    e = a.to("cuda")
    print(f"\nTensor 'e' device: {e.device}")

    # Clear engine cache (if implemented)
    _execution_engine.clear_cache()
