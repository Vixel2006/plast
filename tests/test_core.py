# import ctypes # No longer needed for new design
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

# Import actual classes and functions from your axon library
from plast.core.tensor import Tensor

# from axon.core.buffer import LazyBuffer # No longer needed
# from axon.core.device import Device # Replaced by string device names
# Import specific operations that Tensor uses internally (e.g., for .T)
# These will now be methods on the Tensor class or functions in axon.ops
# from axon.ops.mop import (
#     Transpose,
#     View,
#     Unsqueeze,
#     Squeeze,
#     Expand,
#     Broadcast,
#     Concat,
#     Stack,
# )
# Import binary, unary, and initialization operations
# from axon.ops.bop import Add, Sub, Mul, Div, Pow, MatMul, RSub, RDiv, Dot, Conv2D
# from axon.ops.uop import ReLU, Log, Exp, Abs, Neg, Clip
# from axon.functions import from_data # Replaced by Tensor constructor
# Import new ops functions
from plast.ops import add, sub  # Add more as they are implemented


class TestTensorCore:
    def setup_method(self):
        # self.device = Device("cpu") # Device is now a string
        pass

    def test_tensor_init(self):
        # device = Device("cpu") # Device is now a string
        t = Tensor(shape=(2, 3), device="cpu")  # Use new Tensor constructor
        assert t.shape == (2, 3)
        # assert t.ndim == 2 # ndim is derived from shape
        assert t.device == "cpu"
        # assert t.requires_grad is True # requires_grad not yet implemented in new Tensor
        # assert t.c_tensor_ptr is not None # No longer exists
        # assert t.c_tensor_ptr.contents.grad is not None # No longer exists

    def test_tensor_init_no_grad(self):
        # t = Tensor((2, 2), device="cpu", requires_grad=False) # requires_grad not yet implemented
        t = Tensor(shape=(2, 2), device="cpu")
        # assert t.requires_grad is False
        # assert t.grad is None

    def test_tensor_init_default_device(self):
        # Create a tensor without specifying a device
        t = Tensor(shape=(2, 2))
        # Assert that the device is the default CPU device
        assert t.device == "cpu"
        # assert t.device.index == 0 # No longer exists

    def test_tensor_data_property(self):
        # t = from_data((2, 2), np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        t = Tensor(data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

        expected_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        assert np.array_equal(t.data, expected_data)
        assert t.data.dtype == np.float32

    def test_tensor_shape_property(self):
        # t = Tensor((1, 2, 3), device=self.device)
        t = Tensor(shape=(1, 2, 3), device="cpu")
        assert t.shape == (1, 2, 3)

    # def test_tensor_strides_property(self): # Strides not directly exposed by new Tensor
    #     t_2x3 = Tensor((2, 3), device=self.device)
    #     # Strides for (2,3) should be (3,1)
    #     assert t_2x3.strides == (3, 1)

    #     t_1x2x3 = Tensor((1, 2, 3), device=self.device)
    #     # Strides for (1,2,3) should be (6,3,1)
    #     assert t_1x2x3.strides == (6, 3, 1)

    #     t_4 = Tensor((4,), device=self.device)
    #     # Strides for (4,) should be (1,)
    #     assert t_4.strides == (1,)

    # def test_tensor_ndim_property(self): # ndim is derived from shape
    #     t_1d = Tensor((5,), device=self.device)
    #     assert t_1d.ndim == 1
    #     t_3d = Tensor((2, 3, 4), device=self.device)
    #     assert t_3d.ndim == 3

    # def test_tensor_T_property(self): # Transpose not yet implemented
    #     t = from_data(
    #         (2, 3), np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    #     )

    #     t_T = t.T
    #     # Manually verify transpose output shape and data
    #     expected_transposed_data = np.array(
    #         [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float32
    #     )
    #     assert np.array_equal(t_T.realize().data, expected_transposed_data)

    def test_tensor_numel(self):
        # t = Tensor((2, 3, 4), device=self.device)
        t = Tensor(shape=(2, 3, 4), device="cpu")
        assert t.numel() == 2 * 3 * 4
        # t2 = Tensor((5,), device=self.device)
        t2 = Tensor(shape=(5,), device="cpu")
        assert t2.numel() == 5
        # t3 = Tensor(()) # Scalar tensor
        # assert t3.numel() == 1

    # def test_tensor_del(self): # No longer relevant with shared_ptr
    #     t = Tensor((2, 2), device=self.device)
    #     c_ptr_ref = t.c_tensor_ptr
    #     del t

    # def test_tensor_view(self): # View not yet implemented
    #     a_np = np.arange(12, dtype=np.float32).reshape(3, 4)
    #     a = from_data(a_np.shape, a_np)

    #     b = a.view((4, 3))
    #     assert b.shape == (4, 3)
    #     assert np.array_equal(b.realize().data, a_np.reshape(4, 3))

    #     c = a.view((2, 2, 3))
    #     assert c.shape == (2, 2, 3)
    #     assert np.array_equal(c.realize().data, a_np.reshape(2, 2, 3))

    #     # Test with -1 for inferred dimension
    #     d = a.view((2, -1))
    #     assert d.shape == (2, 6)
    #     assert np.array_equal(d.realize().data, a_np.reshape(2, 6))

    #     with pytest.raises(ValueError, match="Can only specify one -1 in shape."):
    #         a.view((-1, -1))
    #     with pytest.raises(RuntimeError, match="Unable to view as numel mismatch"):
    #         a.view((3, 3))  # 9 elements vs 12

    # def test_tensor_unsqueeze(self): # Unsqueeze not yet implemented
    #     a_np = np.arange(6, dtype=np.float32).reshape(2, 3)
    #     a = from_data(a_np.shape, a_np)

    #     b = a.unsqueeze(0)
    #     assert b.shape == (1, 2, 3)
    #     assert np.array_equal(b.realize().data, a_np.reshape(1, 2, 3))

    #     c = a.unsqueeze(-1)
    #     assert c.shape == (2, 3, 1)
    #     assert np.array_equal(c.realize().data, a_np.reshape(2, 3, 1))

    #     d = a.unsqueeze(1)
    #     assert d.shape == (2, 1, 3)
    #     assert np.array_equal(d.realize().data, a_np.reshape(2, 1, 3))

    #     with pytest.raises(ValueError, match="Dimension 4 out of range"):
    #         a.unsqueeze(4)

    # def test_tensor_squeeze(self): # Squeeze not yet implemented
    #     a_np = np.arange(6, dtype=np.float32).reshape(1, 2, 1, 3, 1)
    #     a = from_data(a_np.shape, a_np)

    #     c = a.squeeze(0)  # Squeeze specific dimension
    #     assert c.shape == (2, 1, 3, 1)
    #     assert np.array_equal(c.realize().data, a_np.reshape(2, 1, 3, 1))

    #     d = a.squeeze(2)  # Squeeze specific dimension
    #     assert d.shape == (1, 2, 3, 1)
    #     assert np.array_equal(d.realize().data, a_np.reshape(1, 2, 3, 1))

    #     e = a.squeeze(-1)  # Squeeze specific dimension (negative index)
    #     assert e.shape == (1, 2, 1, 3)
    #     assert np.array_equal(e.realize().data, a_np.reshape(1, 2, 1, 3))

    #     with pytest.raises(IndexError, match="Dimension out of range"):
    #         a.squeeze(5)
    #     with pytest.raises(IndexError, match="Dimension out of range"):
    #         a.squeeze(-6)

    # def test_tensor_expand(self): # Expand not yet implemented
    #     a_np = np.array([[1, 2, 3]], dtype=np.float32)
    #     a = from_data(a_np.shape, a_np)

    #     b = a.expand((2, 3))
    #     assert b.shape == (2, 3)
    #     expected_b = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
    #     assert np.array_equal(b.realize().data, expected_b)

    #     c_np = np.array([[1], [2]], dtype=np.float32)
    #     c = from_data(c_np.shape, c_np)
    #     d = c.expand((2, 3))
    #     assert d.shape == (2, 3)
    #     expected_d = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)
    #     assert np.array_equal(d.realize().data, expected_d)

    #     # Expand with -1
    #     e = a.expand((2, -1))
    #     assert e.shape == (2, 3)
    #     assert np.array_equal(e.realize().data, expected_b)

    #     with pytest.raises(ValueError, match="Can't expand dimension"):
    #         a.expand((3, 2))  # Cannot expand 3 to 2

    # def test_tensor_broadcast(self): # Broadcast not yet implemented
    #     a_np = np.array([[1, 2, 3]], dtype=np.float32)
    #     a = from_data(a_np.shape, a_np)

    #     b = a.broadcast((2, 3))
    #     assert b.shape == (2, 3)
    #     expected_b = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
    #     assert np.array_equal(b.realize().data, expected_b)

    #     c_np = np.array([[1], [2]], dtype=np.float32)
    #     c = from_data(c_np.shape, c_np)
    #     d = c.broadcast((2, 3))
    #     assert d.shape == (2, 3)
    #     expected_d = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)
    #     assert np.array_equal(d.realize().data, expected_d)

    #     # Test with incompatible shapes
    #     with pytest.raises(ValueError, match="Shapes are not broadcastable"):
    #         a.broadcast(
    #             (3, 2)
    #         )  # (1,3) to (3,2) is not directly broadcastable without more complex rules

    # def test_tensor_detach(self): # Detach not yet implemented
    #     a = from_data((2, 2), [[1, 2], [3, 4]])
    #     b = a + 1.0
    #     b.realize()
    #     assert b._lazy_buffer is not None
    #     assert b.requires_grad is True

    #     c = b.detach()
    #     assert c is not b
    #     assert np.array_equal(c.data, b.data)  # Data should be the same
    #     assert c._lazy_buffer is None  # Lazy buffer should be cleared
    #     assert (
    #         c.requires_grad is False
    #     )  # requires_grad should be False for detached tensor

    #     # Ensure that detaching breaks the graph for backward pass
    #     with pytest.raises(RuntimeError, match="Gradient storage not allocated"):
    #         c.backward()  # Should fail because the graph is broken

    #     # Original tensor 'a' should still have its graph intact
    #     a_orig = from_data(
    #         (2, 2), np.ones((2, 2), dtype=np.float32), requires_grad=True
    #     )
    #     d = a_orig + 1.0
    #     d.backward()
    #     assert np.all(a_orig.grad.data == 1.0)  # Should still work

    # def test_tensor_exp(self): # Exp not yet implemented
    #     a_np = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    #     a = from_data(a_np.shape, a_np)

    #     b = a.exp()
    #     expected_b = np.exp(a_np)
    #     assert np.allclose(b.realize().data, expected_b)

    #     # Test backward
    #     b.backward()
    #     expected_a_grad = np.exp(a_np)  # d(exp(x))/dx = exp(x)
    #     assert np.allclose(a.grad.data, expected_a_grad)

    # def test_tensor_log(self): # Log not yet implemented
    #     a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    #     a = from_data(a_np.shape, a_np)

    #     b = a.log()
    #     expected_b = np.log(a_np)
    #     assert np.allclose(b.realize().data, expected_b)

    #     # Test backward
    #     b.backward()
    #     expected_a_grad = 1.0 / a_np  # d(log(x))/dx = 1/x
    #     assert np.allclose(a.grad.data, expected_a_grad)

    # def test_tensor_abs(self): # Abs not yet implemented
    #     a_np = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    #     a = from_data(a_np.shape, a_np)

    #     b = a.abs()
    #     expected_b = np.abs(a_np)
    #     assert np.allclose(b.realize().data, expected_b)

    #     # Test backward
    #     b.backward()
    #     expected_a_grad = np.sign(a_np)  # d(|x|)/dx = sign(x)
    #     assert np.allclose(a.grad.data, expected_a_grad)

    # def test_tensor_relu(self): # ReLU not yet implemented
    #     a_np = np.array([[-1.0, 0.0], [2.0, -3.0]], dtype=np.float32)
    #     a = from_data(a_np.shape, a_np)

    #     b = a.relu()
    #     expected_b = np.maximum(0, a_np)
    #     assert np.allclose(b.realize().data, expected_b)

    #     # Test backward
    #     b.backward()
    #     expected_a_grad = (a_np > 0).astype(
    #         np.float32
    #     )  # d(relu(x))/dx = 1 if x > 0 else 0
    #     assert np.allclose(a.grad.data, expected_a_grad)

    # def test_tensor_neg(self): # Neg not yet implemented
    #     a_np = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    #     a = from_data(a_np.shape, a_np)

    #     b = -a
    #     expected_b = -a_np
    #     assert np.allclose(b.realize().data, expected_b)

    #     # Test backward
    #     b.backward()
    #     expected_a_grad = np.array([[-1.0, -1.0], [-1.0, -1.0]], dtype=np.float32)
    #     assert np.allclose(a.grad.data, expected_a_grad)

    def test_tensor_add(self):
        # a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        # b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        # a = from_data(a_np.shape, a_np)
        # b = from_data(b_np.shape, b_np)
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        a = Tensor(data=a_np)
        b = Tensor(data=b_np)

        # Tensor + Tensor
        c = a + b
        expected_c = a_np + b_np
        # assert np.allclose(c.realize().data, expected_c)
        assert np.allclose(c.data, expected_c)
        # c.backward() # Autograd not yet implemented
        # assert np.allclose(a.grad.data, np.ones_like(a_np))
        # assert np.allclose(b.grad.data, np.ones_like(b_np))

        # Tensor + scalar
        d = a + 10.0
        expected_d = a_np + 10.0
        # assert np.allclose(d.realize().data, expected_d)
        assert np.allclose(d.data, expected_d)
        # d.backward() # Autograd not yet implemented
        # assert np.allclose(
        #     a.grad.data, np.ones_like(a_np) * 2
        # )  # Accumulated from previous backward

        # scalar + Tensor (__radd__)
        e_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        # e = from_data(e_np.shape, e_np)
        e = Tensor(data=e_np)
        f = 20.0 + e
        expected_f = 20.0 + e_np
        # assert np.allclose(f.realize().data, expected_f)
        assert np.allclose(f.data, expected_f)
        # f.backward() # Autograd not yet implemented
        # assert np.allclose(e.grad.data, np.ones_like(e_np) * 20.0)

    def test_tensor_sub(self):
        # a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        # b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        # a = from_data(a_np.shape, a_np)
        # b = from_data(b_np.shape, b_np)
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        a = Tensor(data=a_np)
        b = Tensor(data=b_np)

        # Tensor - Tensor
        c = a - b
        expected_c = a_np - b_np
        # assert np.allclose(c.realize().data, expected_c)
        assert np.allclose(c.data, expected_c)
        # c.backward() # Autograd not yet implemented
        # assert np.allclose(a.grad.data, np.ones_like(a_np))
        # assert np.allclose(b.grad.data, -np.ones_like(b_np))

        # Tensor - scalar
        d = a - 10.0
        expected_d = a_np - 10.0
        # assert np.allclose(d.realize().data, expected_d)
        assert np.allclose(d.data, expected_d)
        # d.backward() # Autograd not yet implemented
        # assert np.allclose(
        #     a.grad.data, np.ones_like(a_np) * 2
        # )  # Accumulated from previous backward

        # scalar - Tensor (__rsub__)
        e_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        # e = from_data(e_np.shape, e_np)
        e = Tensor(data=e_np)
        f = 20.0 - e
        expected_f = 20.0 - e_np
        # assert np.allclose(f.realize().data, expected_f)
        assert np.allclose(f.data, expected_f)
        # f.backward() # Autograd not yet implemented
        # assert np.allclose(e.grad.data, -np.ones_like(e_np))

    # def test_tensor_mul(self): # Mul not yet implemented
    #     a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    #     b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    #     a = from_data(a_np.shape, a_np)
    #     b = from_data(b_np.shape, b_np)

    #     # Tensor * Tensor
    #     c = a * b
    #     expected_c = a_np * b_np
    #     assert np.allclose(c.realize().data, expected_c)
    #     c.backward()
    #     assert np.allclose(a.grad.data, b_np)
    #     assert np.allclose(b.grad.data, a_np)

    #     # Tensor * scalar
    #     d = a * 10.0
    #     expected_d = a_np * 10.0
    #     assert np.allclose(d.realize().data, expected_d)
    #     d.backward()
    #     assert np.allclose(a.grad.data, b_np + 10.0)  # Accumulated

    #     # scalar * Tensor (__rmul__)
    #     e_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    #     e = from_data(e_np.shape, e_np)
    #     f = 20.0 * e
    #     expected_f = 20.0 * e_np
    #     assert np.allclose(f.realize().data, expected_f)
    #     f.backward()
    #     assert np.allclose(e.grad.data, np.ones_like(e_np) * 20.0)

    # def test_tensor_div(self): # Div not yet implemented
    #     a_np = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    #     b_np = np.array([[5.0, 4.0], [3.0, 2.0]], dtype=np.float32)
    #     a = from_data(a_np.shape, a_np)
    #     b = from_data(b_np.shape, b_np)

    #     # Tensor / Tensor
    #     c = a / b
    #     expected_c = a_np / b_np
    #     assert np.allclose(c.realize().data, expected_c)
    #     c.backward()
    #     assert np.allclose(a.grad.data, 1.0 / b_np)
    #     assert np.allclose(b.grad.data, -a_np / (b_np**2))

    #     # Tensor / scalar
    #     d = a / 2.0
    #     expected_d = a_np / 2.0
    #     assert np.allclose(d.realize().data, expected_d)
    #     d.backward()
    #     assert np.allclose(a.grad.data, (1.0 / b_np) + (1.0 / 2.0))  # Accumulated

    #     # scalar / Tensor (__rtruediv__)
    #     e_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    #     e = from_data(e_np.shape, e_np)
    #     f = 20.0 / e
    #     expected_f = 20.0 / e_np
    #     assert np.allclose(f.realize().data, expected_f)
    #     f.backward()
    #     assert np.allclose(e.grad.data, -20.0 / (e_np**2))

    # def test_tensor_pow(self): # Pow not yet implemented
    #     a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    #     b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    #     a = from_data(a_np.shape, a_np)
    #     b = from_data(b_np.shape, b_np)

    #     # Tensor ** Tensor
    #     c = a**b
    #     expected_c = a_np**b_np
    #     assert np.allclose(c.realize().data, expected_c)
    #     c.backward()
    #     # d(a^b)/da = b * a^(b-1)
    #     # d(a^b)/db = a^b * log(a)
    #     assert np.allclose(a.grad.data, b_np * (a_np ** (b_np - 1)))
    #     assert np.allclose(b.grad.data, (a_np**b_np) * np.log(a_np))

    #     # Tensor ** scalar
    #     d = a**3.0
    #     expected_d = a_np**3.0
    #     assert np.allclose(d.realize().data, expected_d)
    #     d.backward()
    #     assert np.allclose(
    #         a.grad.data, (b_np * (a_np ** (b_np - 1))) + (3.0 * (a_np**2))
    #     )  # Accumulated

    # def test_tensor_matmul(self): # Matmul not yet implemented
    #     # Matrix-Matrix multiplication
    #     a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    #     b_np = np.array([[5, 6], [7, 8]], dtype=np.float32)
    #     a = from_data(a_np.shape, a_np)
    #     b = from_data(b_np.shape, b_np)

    #     c = a @ b
    #     expected_c = a_np @ b_np
    #     assert np.allclose(c.realize().data, expected_c, rtol=1e-4, atol=1e-4)
    #     c.backward()
    #     # dL/dA = dL/dC @ B.T
    #     # dL/dB = A.T @ dL/dC
    #     assert np.allclose(
    #         a.grad.data, np.ones_like(expected_c) @ b_np.T, rtol=1e-4, atol=1e-4
    #     )
    #     assert np.allclose(
    #         b.grad.data, a_np.T @ np.ones_like(expected_c), rtol=1e-4, atol=1e-4
    #     )

    #     # Batch matrix multiplication (example: (2, 2, 3) @ (2, 3, 2))
    #     j_np = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    #     k_np = np.arange(12, 24, dtype=np.float32).reshape(2, 3, 2)
    #     j = from_data(j_np.shape, j_np)
    #     k = from_data(k_np.shape, k_np)

    #     l = j @ k
    #     expected_l = j_np @ k_np
    #     assert np.allclose(l.realize().data, expected_l, rtol=1e-4, atol=1e-4)
    #     l.backward()
    #     # For batch matmul, gradients are also batch-wise
    #     # dL/dJ = dL/dL @ K.T (batch-wise)
    #     # dL/dK = J.T @ dL/dL (batch-wise)
    #     # np.einsum can help here for batch matmul gradients
    #     expected_j_grad = np.einsum(
    #         "...ij,...jk->...ik", np.ones_like(expected_l), np.swapaxes(k_np, -1, -2)
    #     )
    #     expected_k_grad = np.einsum(
    #         "...ki,...il->...kl", np.swapaxes(j_np, -1, -2), np.ones_like(expected_l)
    #     )
    #     assert np.allclose(j.grad.data, expected_j_grad, rtol=1e-4, atol=1e-4)
    #     assert np.allclose(k.grad.data, expected_k_grad, rtol=1e-4, atol=1e-4)


# Remove TestLazyBufferCore entirely
# class TestLazyBufferCore:
#     def setup_method(self):
#         self.device = Device("cpu")

#     def test_lazy_buffer_init(self):
#         out_tensor = Tensor((2, 2), device=self.device)
#         mock_op = Add()  # Use a real op
#         prev_tensors = [Tensor((2, 2), device=self.device)]
#         forward_kwargs = {"scalar_val": 5.0}
#         backward_ctx = ctypes.cast(ctypes.pointer(ctypes.c_float(5.0)), ctypes.c_void_p)

#         lb = LazyBuffer(
#             out=out_tensor,
#             op=mock_op,
#             prev=prev_tensors,
#             forward_kwargs=forward_kwargs,
#             backward_ctx=backward_ctx,
#         )

#         assert lb.out is out_tensor
#         assert lb.op is mock_op
#         assert lb.prev == prev_tensors
#         assert lb.forward_kwargs == forward_kwargs
#         assert lb.backward_ctx == backward_ctx
#         assert not lb._realized
#         assert lb._topo_sorted is None

#     def test_lazy_buffer_topo_sort_simple(self):
#         a = Tensor((2, 2), device=self.device, requires_grad=True)
#         b = Tensor((2, 2), device=self.device, requires_grad=True)
#         c = a + b  # c = Add.create_node(a, b)
#         d = c.relu()  # d = ReLU.create_node(c)

#         # The topo sort for d should process dependencies before d itself.
#         # The result should contain c's LazyBuffer then d's LazyBuffer.
#         sorted_buffers = d._lazy_buffer.topo_sort()

#         assert len(sorted_buffers) == 2
#         assert c._lazy_buffer in sorted_buffers
#         assert d._lazy_buffer in sorted_buffers
#         assert sorted_buffers.index(c._lazy_buffer) < sorted_buffers.index(
#             d._lazy_buffer
#         )

#     def test_lazy_buffer_topo_sort_complex(self):
#         a = Tensor((2, 2), device=self.device, requires_grad=True)
#         b = Tensor((2, 2), device=self.device, requires_grad=True)
#         c = a * b
#         d = c + a  # 'd' depends on 'c' and 'a'. 'c' depends on 'a' and 'b'.
#         e = d.exp()

#         sorted_buffers = e._lazy_buffer.topo_sort()

#         assert len(sorted_buffers) == 3  # c_lb, d_lb, e_lb
#         assert c._lazy_buffer in sorted_buffers
#         assert d._lazy_buffer in sorted_buffers
#         assert e._lazy_buffer in sorted_buffers

#         # Order must be dependencies first
#         # assert sorted_buffers.index(c._lazy_buffer) < sorted_buffers.index(d._lazy_buffer)
#         # assert sorted_buffers.index(d._lazy_buffer) < sorted_buffers.index(e._lazy_buffer)

#         # `a` and `b` are base Tensors, not LazyBuffers, so they won't appear in `sorted_buffers` list.
#         # But their realization (or access to their `c_tensor_ptr`) is implicit in `c` and `d` ops.

#     def test_lazy_buffer_topo_sort_circular_dependency(self):
#         # This scenario should generally be prevented by the design of LazyOp.create_node
#         # which creates a new output tensor. However, if somehow manually created or
#         # through a bug, a circular dependency could exist.

#         # To simulate a circular dependency for this test:
#         t1 = Tensor((1,), device=self.device, requires_grad=True)
#         t2 = Tensor((1,), device=self.device, requires_grad=True)

#         # Manually create LazyBuffers and enforce a cycle for testing topo_sort's detection
#         op1 = Add()
#         op2 = Add()

#         # Create lazy_buffer objects
#         lb1 = LazyBuffer(t1, op1, [], {}, None)  # Will temporarily link prev later
#         lb2 = LazyBuffer(t2, op2, [], {}, None)  # Will temporarily link prev later

#         # Assign lazy buffers to tensors
#         t1._lazy_buffer = lb1
#         t2._lazy_buffer = lb2

#         # Now create the circular dependency
#         lb1.prev = [t2]  # t1 depends on t2
#         lb2.prev = [t1]  # t2 depends on t1

#         with pytest.raises(RuntimeError, match="Circular dependency detected"):
#             t1._lazy_buffer.topo_sort()

#         # Clean up to avoid issues with subsequent tests if not properly isolated
#         t1._lazy_buffer = None
#         t2._lazy_buffer = None

#     def test_lazy_buffer_realize_and_forward(self):
#         a_flat_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
#         b_flat_data = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
#         a = from_data((2, 2), a_flat_data.reshape(2, 2), requires_grad=True)
#         b = from_data((2, 2), b_flat_data.reshape(2, 2), requires_grad=True)

#         c = a + b  # This creates c with a LazyBuffer for Add operation
#         d = c.relu()  # This creates d with a LazyBuffer for ReLU operation

#         assert not d._lazy_buffer._realized
#         assert not c._lazy_buffer._realized

#         realized_d = d.realize()  # This should trigger forward for c then d

#         assert realized_d is d  # realize returns the final output tensor
#         assert d._lazy_buffer._realized
#         assert c._lazy_buffer._realized

#         # Verify the data after realization
#         expected_c_data = a_flat_data + b_flat_data  # [6.0, 8.0, 10.0, 12.0]
#         expected_d_data = np.maximum(expected_c_data, 0)  # ReLU, so same as c_data here

#         assert np.allclose(c.data.flatten(), expected_c_data, rtol=1e-4, atol=1e-4)
#         assert np.allclose(d.data.flatten(), expected_d_data, rtol=1e-4, atol=1e-4)

#     def test_lazy_buffer_backward(self):
#         # Test a simple graph: a -> add_scalar -> c
#         a_flat_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
#         a = from_data((2, 2), a_flat_data.reshape(2, 2), requires_grad=True)

#         scalar_val = 5.0
#         c = a + scalar_val  # c = Add.create_node(a, scalar_val)

#         # Trigger backward pass from 'c'
#         c.backward()

#         # After c.backward():
#         # 1. c.realize() is called (which calls Add.forward)
#         # 2. c._lazy_buffer.topo_sort() is called (reversed)
#         # 3. For 'c' (the last node in topological order), its grad is initialized to ones.
#         # 4. Add.backward is called.

#         # Verify c.grad (should be all ones as it's the final output of the backward call)
#         assert np.allclose(c.grad.data, 1.0, rtol=1e-4, atol=1e-4)

#         # Verify a.grad
#         # For c = a + S, dc/da = 1. So, gradient of a should be the gradient of c.
#         # The `c_add_grad_op` should propagate `out_grad_ptr` to the `prev_ptrs`.
#         # Assuming `c_add_grad_op` correctly copies the gradient:
#         assert np.allclose(a.grad.data, 1.0, rtol=1e-4, atol=1e-4)

#     def test_lazy_buffer_backward_complex_graph(self):
#         # Graph: a --- (mul) ---> c --- (relu) ---> d
#         #        |                    ^
#         #        +--------- (add) ---+
#         a_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
#         b_data = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
#         a = from_data(a_data.shape, a_data, requires_grad=True)
#         b = from_data(b_data.shape, b_data, requires_grad=True)

#         c_mul = a * b  # Add.create_node(a, b)
#         c_add = (
#             c_mul + a
#         )  # Mul.create_node(c_mul, a) (assuming add_grad accumulates correctly)
#         d = c_add.relu()  # ReLU.create_node(c_add)

#         # Execute backward from 'd'
#         d.backward()

#         # Expected values (manual calculation for verification)
#         # d = ReLU(c_add)
#         # c_add = (a * b) + a
#         #
#         # ∂L/∂d = 1 (initialized for d)
#         #
#         # ∂L/∂c_add = ∂L/∂d * ∂d/∂c_add = 1 * (1 if c_add > 0 else 0)
#         # Since all input values are positive, c_mul > 0, c_add > 0. So ∂d/∂c_add = 1.
#         # Thus, ∂L/∂c_add = 1.0
#         #
#         # For c_add = (a * b) + a:
#         # ∂c_add/∂(a*b) = 1
#         # ∂c_add/∂a = 1 (from the direct 'a' input to the sum)
#         #
#         # So, grad of c_mul should be ∂L/∂c_add * ∂c_add/∂(a*b) = 1 * 1 = 1.0
#         # And additional grad for 'a' from this path should be ∂L/∂c_add * ∂c_add/∂a = 1 * 1 = 1.0
#         #
#         # For c_mul = a * b:
#         # ∂c_mul/∂a = b
#         # ∂c_mul/∂b = a
#         #
#         # Total grad for 'a':
#         # ∂L/∂a = (∂L/∂c_add * ∂c_add/∂a) + (∂L/∂c_mul * ∂c_mul/∂a)
#         # ∂L/∂a = (1 * 1) + (1 * b) = 1 + b
#         #
#         # Total grad for 'b':
#         # ∂L/∂b = (∂L/∂c_mul * ∂c_mul/∂b)
#         # ∂L/∂b = 1 * a = a

#         # Calculate expected gradients
#         # c_mul_val = a_data * b_data = [[0.5, 1.0], [1.5, 2.0]]
#         # c_add_val = c_mul_val + a_data = [[1.5, 3.0], [4.5, 6.0]] (all > 0)

#         expected_d_grad = np.ones_like(d.data)
#         expected_c_add_grad = expected_d_grad * (c_add.data > 0)  # ReLU gradient

#         expected_a_grad = expected_c_add_grad + (
#             expected_c_add_grad * b.data
#         )  # dL/dc_add * dc_add/da + dL/dcmul * dcmul/da
#         expected_b_grad = expected_c_add_grad * a.data  # dL/dc_mul * dc_mul/db

#         # Due to potential floating point inaccuracies from C operations, use `np.isclose`
#         assert np.allclose(d.grad.data, expected_d_grad, rtol=1e-4, atol=1e-4)
#         assert np.allclose(a.grad.data, expected_a_grad, rtol=1e-4, atol=1e-4)
#         assert np.allclose(b.grad.data, expected_b_grad, rtol=1e-4, atol=1e-4)
