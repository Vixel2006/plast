from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

import plast._plast_cpp_core as _plast_cpp_core

from ..core.device import Device, get_default_device

# Map Python types to C++ DTypes
_DTYPE_MAP = {
    np.float32: _plast_cpp_core.DType.FLOAT32,
    np.float64: _plast_cpp_core.DType.FLOAT64,
    np.int8: _plast_cpp_core.DType.INT8, np.int16: _plast_cpp_core.DType.INT16,
    np.int32: _plast_cpp_core.DType.INT32,
    np.int64: _plast_cpp_core.DType.INT64,
    np.uint8: _plast_cpp_core.DType.UINT8,
    np.uint16: _plast_cpp_core.DType.UINT16,
    np.uint32: _plast_cpp_core.DType.UINT32,
    np.uint64: _plast_cpp_core.DType.UINT64,
    np.bool_: _plast_cpp_core.DType.BOOL,
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

            # Determine device
            if device is None:
                actual_device = get_default_device()
            else:
                actual_device = Device.parse(device)
            cpp_device = actual_device.cpp_device_type

            # Create a C++ Tensor object from numpy data using from_data
            cpp_tensor_value = _plast_cpp_core.from_data(
                data, list(shape), cpp_dtype, cpp_device
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

            # Determine device
            if device is None:
                actual_device = get_default_device()
            else:
                actual_device = Device.parse(device)
            cpp_device = actual_device.cpp_device_type

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

        # Use the new pybind11 method to get data as a numpy array
        return cpp_tensor._get_data_as_numpy()

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

    @property
    def T(self) -> Tensor:
        if len(self.shape) < 2:
            raise ValueError("Transpose requires at least 2 dimensions.")
        N = len(self.shape) - 2
        M = len(self.shape) - 1
        new_cpp_node = _plast_cpp_core.transpose_op_node(self._cpp_node, N, M)
        return Tensor(cpp_node=new_cpp_node)

    def to(self, target_device_str: str) -> Tensor:
        # Execute the graph up to this node to get the concrete C++ Tensor
        cpp_tensor_value = _execution_engine.execute(self._cpp_node)

        # Parse the target device string to get the C++ DeviceType
        target_device = Device.parse(target_device_str)
        target_cpp_device = target_device.cpp_device_type

        # Call the C++ Tensor's to() method for device transfer
        new_cpp_tensor_value = cpp_tensor_value.to(target_cpp_device)

        # Create a new C++ Node wrapping the new C++ Tensor
        new_cpp_node = _plast_cpp_core.Node(new_cpp_tensor_value)

        # Return a new Python Tensor instance wrapping the new C++ Node
        return Tensor(cpp_node=new_cpp_node)

    def __add__(self, other: Tensor | float | int) -> Tensor:
        if isinstance(other, Tensor):
            new_cpp_node = _plast_cpp_core.add_op_node(self._cpp_node, other._cpp_node)
            return Tensor(cpp_node=new_cpp_node)
        elif isinstance(other, (int, float)):
            scalar_tensor = Tensor(data=np.array(other, dtype=np.float32))
            new_cpp_node = _plast_cpp_core.add_op_node(
                self._cpp_node, scalar_tensor._cpp_node
            )
            return Tensor(cpp_node=new_cpp_node)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for +: 'Tensor' and '{type(other)}'"
            )

    def __radd__(self, other: float | int) -> Tensor:
        return self.__add__(other)

    def __sub__(self, other: Tensor | float | int) -> Tensor:
        if isinstance(other, Tensor):
            new_cpp_node = _plast_cpp_core.sub_op_node(self._cpp_node, other._cpp_node)
            return Tensor(cpp_node=new_cpp_node)
        elif isinstance(other, (int, float)):
            scalar_tensor = Tensor(data=np.array(other, dtype=np.float32))
            new_cpp_node = _plast_cpp_core.sub_op_node(
                self._cpp_node, scalar_tensor._cpp_node
            )
            return Tensor(cpp_node=new_cpp_node)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for -: 'Tensor' and '{type(other)}'"
            )

    def __rsub__(self, other: float | int) -> Tensor:
        # For rsub (e.g., 5 - Tensor), it's equivalent to Tensor(5) - self
        scalar_tensor = Tensor(data=np.array(other, dtype=np.float32))
        new_cpp_node = _plast_cpp_core.sub_op_node(
            scalar_tensor._cpp_node, self._cpp_node
        )
        return Tensor(cpp_node=new_cpp_node)

    def __mul__(self, other: Tensor | float | int) -> Tensor:
        if isinstance(other, Tensor):
            new_cpp_node = _plast_cpp_core.mul_op_node(self._cpp_node, other._cpp_node)
            return Tensor(cpp_node=new_cpp_node)
        elif isinstance(other, (int, float)):
            scalar_tensor = Tensor(data=np.array(other, dtype=np.float32))
            new_cpp_node = _plast_cpp_core.mul_op_node(
                self._cpp_node, scalar_tensor._cpp_node
            )
            return Tensor(cpp_node=new_cpp_node)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for -: 'Tensor' and '{type(other)}'"
            )

    def __matmul__(self, other: Tensor) -> Tensor:
        new_cpp_node = _plast_cpp_core.matmul_op_node(self._cpp_node, other._cpp_node)
        return Tensor(cpp_node=new_cpp_node)

    def abs(self) -> Tensor:
        new_cpp_node = _plast_cpp_core.abs_op_node(self._cpp_node)
        return Tensor(cpp_node=new_cpp_node)

    def relu(self) -> Tensor:
        new_cpp_node = _plast_cpp_core.relu_op_node(self._cpp_node)
        return Tensor(cpp_node=new_cpp_node)

    def lrelu(self, alpha: float) -> Tensor:
        new_cpp_node = _plast_cpp_core.leaky_relu_op_node(self._cpp_node, alpha)
        return Tensor(cpp_node=new_cpp_node)

    def exp(self) -> Tensor:
        new_cpp_node = _plast_cpp_core.exp_op_node(self._cpp_node)
        return Tensor(cpp_node=new_cpp_node)

    def log(self) -> Tensor:
        new_cpp_node = _plast_cpp_core.log_op_node(self._cpp_node)
        return Tensor(cpp_node=new_cpp_node)

    def reshape(self, new_shape: Tuple[int, ...]) -> Tensor:
        new_cpp_node = _plast_cpp_core.view_op_node(self._cpp_node, list(new_shape))
        return Tensor(cpp_node=new_cpp_node)

    def view(self, *shape: int) -> Tensor:
        if not shape:
            raise ValueError("View operation requires at least one dimension.")
        new_shape = list(shape)
        new_cpp_node = _plast_cpp_core.view_op_node(self._cpp_node, new_shape)
        return Tensor(cpp_node=new_cpp_node)

    def transpose(self, N: int, M: int) -> Tensor:
        new_cpp_node = _plast_cpp_core.transpose_op_node(self._cpp_node, N, M)
        return Tensor(cpp_node=new_cpp_node)

    def squeeze(self, dim: Optional[int] = None) -> Tensor:
        if dim is None:
            # Squeeze all dimensions of size 1
            current_shape = list(self.shape)
            squeezed_dims = []
            for i, s in enumerate(current_shape):
                if s == 1:
                    squeezed_dims.append(i)
            
            if not squeezed_dims:
                return self # No dimensions to squeeze

            # Apply squeeze iteratively for each dimension of size 1
            result_tensor = self
            # Squeeze from highest dimension to lowest to avoid index shifts
            for d in sorted(squeezed_dims, reverse=True):
                result_tensor = Tensor(cpp_node=_plast_cpp_core.squeeze_op_node(result_tensor._cpp_node, d, d)) # N and M are the same for squeeze
            return result_tensor
        else:
            if dim < 0:
                dim += len(self.shape)
            if not (0 <= dim < len(self.shape)):
                raise IndexError("Dimension out of range.")
            if self.shape[dim] != 1:
                return self # Cannot squeeze dimension of size > 1
            new_cpp_node = _plast_cpp_core.squeeze_op_node(self._cpp_node, dim, dim) # N and M are the same for squeeze
            return Tensor(cpp_node=new_cpp_node)

    def unsqueeze(self, dim: int) -> Tensor:
        if dim < 0:
            dim += len(self.shape) + 1 # +1 because we are adding a dimension
        if not (0 <= dim <= len(self.shape)):
            raise IndexError("Dimension out of range.")
        new_cpp_node = _plast_cpp_core.unsqueeze_op_node(self._cpp_node, dim)
        return Tensor(cpp_node=new_cpp_node)

    def expand(self, *new_shape: int) -> Tensor:
        if not new_shape:
            raise ValueError("Expand operation requires at least one dimension.")
        
        current_tensor = self
        target_ndim = len(new_shape)
        current_ndim = len(self.shape)

        if target_ndim < current_ndim:
            raise ValueError(
                f"Cannot expand tensor from {current_ndim} dimensions to {target_ndim} dimensions. "
                "Expand can only add new dimensions of size 1 or expand existing singleton dimensions."
            )
        
        # Add new dimensions of size 1 at the front if target_ndim > current_ndim
        for _ in range(target_ndim - current_ndim):
            current_tensor = current_tensor.unsqueeze(0)
        
        # Now, current_tensor has the same number of dimensions as new_shape.
        # We can use broadcast_to to expand existing singleton dimensions.
        return current_tensor.broadcast_to(*new_shape)

    def broadcast_to(self, *target_shape: int) -> Tensor:
        if not target_shape:
            raise ValueError("Broadcast operation requires at least one dimension.")
        new_cpp_node = _plast_cpp_core.broadcast_op_node(self._cpp_node, list(target_shape))
        return Tensor(cpp_node=new_cpp_node)

    def min(self, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        if dim is None:
            new_cpp_node = _plast_cpp_core.min_op_node_full(self._cpp_node)
        else:
            new_cpp_node = _plast_cpp_core.min_op_node_dim(self._cpp_node, dim, keepdim)
        return Tensor(cpp_node=new_cpp_node)

    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        if dim is None:
            new_cpp_node = _plast_cpp_core.mean_op_node_full(self._cpp_node)
        else:
            new_cpp_node = _plast_cpp_core.mean_op_node_dim(self._cpp_node, dim, keepdim)
        return Tensor(cpp_node=new_cpp_node)

    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        if dim is None:
            new_cpp_node = _plast_cpp_core.sum_op_node_full(self._cpp_node)
        else:
            new_cpp_node = _plast_cpp_core.sum_op_node_dim(self._cpp_node, dim, keepdim)
        return Tensor(cpp_node=new_cpp_node)

    def max(self, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        if dim is None:
            new_cpp_node = _plast_cpp_core.max_op_node_full(self._cpp_node)
        else:
            new_cpp_node = _plast_cpp_core.max_op_node_dim(self._cpp_node, dim, keepdim)
        return Tensor(cpp_node=new_cpp_node)

    def __repr__(self) -> str:
        # For repr, we might not want to trigger full execution.
        # This would require the C++ Node to expose its inferred shape/dtype without execution.
        # For now, we'll execute to get details.
        try:
            cpp_tensor = _execution_engine.execute(self._cpp_node)
            return f"Tensor(shape={tuple(cpp_tensor.shape)}, dtype={_REVERSE_DTYPE_MAP.get(cpp_tensor.dtype)}, device={cpp_tensor.device})"
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
    # Create some tensors on CPU
    a_cpu = Tensor(data=[[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, device="cpu")
    b_cpu = Tensor(data=[[[5.0, 6.0], [7.0, 8.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32, device="cpu")

    # Perform an operation on CPU
    c_cpu = a_cpu + b_cpu

    # Access data (triggers execution)
    print("Result of a_cpu + b_cpu:")
    print(c_cpu.data)
    print(f"Shape: {c_cpu.shape}, DType: {c_cpu.dtype}, Device: {c_cpu.device}")
    
    # Example of expand: expand (2,2) to (1,2,2)
    print(a_cpu.max().data)

