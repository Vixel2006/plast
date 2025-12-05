from __future__ import annotations

import numpy as np

from plast.core.tensor import Tensor

from .. import _plast_cpp_core


# Binary Operations
def add(lhs: Tensor, rhs: Tensor | float) -> Tensor:
    if isinstance(rhs, Tensor):
        new_cpp_node = _plast_cpp_core.add_op_node(lhs._cpp_node, rhs._cpp_node)
        return Tensor(cpp_node=new_cpp_node)
    elif isinstance(rhs, (int, float)):
        scalar_tensor = Tensor(
            data=np.array(rhs, dtype=np.float32)
        )  # Assuming float32 for scalar
        new_cpp_node = _plast_cpp_core.add_op_node(
            lhs._cpp_node, scalar_tensor._cpp_node
        )
        return Tensor(cpp_node=new_cpp_node)
    else:
        raise TypeError(
            f"Unsupported operand type(s) for add: 'Tensor' and '{type(rhs)}'"
        )


def sub(lhs: Tensor, rhs: Tensor | float) -> Tensor:
    if isinstance(rhs, Tensor):
        new_cpp_node = _plast_cpp_core.sub_op_node(lhs._cpp_node, rhs._cpp_node)
        return Tensor(cpp_node=new_cpp_node)
    elif isinstance(rhs, (int, float)):
        scalar_tensor = Tensor(
            data=np.array(rhs, dtype=np.float32)
        )  # Assuming float32 for scalar
        new_cpp_node = _plast_cpp_core.sub_op_node(
            lhs._cpp_node, scalar_tensor._cpp_node
        )
        return Tensor(cpp_node=new_cpp_node)
    else:
        raise TypeError(
            f"Unsupported operand type(s) for -: 'Tensor' and '{type(rhs)}'"
        )

def matmul(lhs: Tensor, rhs: Tensor) -> Tensor:
    new_cpp_node = _plast_cpp_core.matmul_op_node(lhs._cpp_node, rhs._cpp_node)
    return Tensor(cpp_node=new_cpp_node)


__all__ = [
    "add",
    "sub",
    "matmul",
]
