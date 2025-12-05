import pytest
import numpy as np
import time

from plast.core.tensor import Tensor
from plast.core.device import Device

# Helper function to compare Tensors with numpy arrays
def assert_tensor_equal(tensor: Tensor, np_array: np.ndarray):
    assert tensor.shape == np_array.shape
    assert tensor.dtype == np_array.dtype.type
    np.testing.assert_allclose(tensor.data, np_array, rtol=1e-5, atol=1e-8)

# Helper function to create a tensor on a specific device
def create_tensor(data, dtype=np.float32, device="cpu"):
    return Tensor(data=data, dtype=dtype, device=device)

# Fixture for common tensor data
@pytest.fixture
def sample_tensor_data():
    return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

@pytest.fixture
def sample_tensor(sample_tensor_data):
    return Tensor(data=sample_tensor_data)

@pytest.fixture
def sample_tensor_int_data():
    return np.array([[1, 2], [3, 4]], dtype=np.int32)

@pytest.fixture
def sample_tensor_int(sample_tensor_int_data):
    return Tensor(data=sample_tensor_int_data)

@pytest.fixture
def broadcastable_tensor_data():
    return np.array([5.0, 6.0], dtype=np.float32)

@pytest.fixture
def broadcastable_tensor(broadcastable_tensor_data):
    return Tensor(data=broadcastable_tensor_data)

@pytest.fixture
def large_tensor_data():
    return np.arange(1, 28, dtype=np.float32).reshape(3, 3, 3)

@pytest.fixture
def large_tensor(large_tensor_data):
    return Tensor(data=large_tensor_data)


# Test Tensor initialization
class TestTensorInitialization:
    def test_init_from_numpy_array(self):
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        t = Tensor(data=data)
        assert_tensor_equal(t, data)
        assert t.dtype == np.float32
        assert t.device == "cpu"

    def test_init_from_list(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        t = Tensor(data=data, dtype=np.float32)
        assert_tensor_equal(t, np.array(data, dtype=np.float32))

    def test_init_from_tuple(self):
        data = ((1, 2), (3, 4))
        t = Tensor(data=data, dtype=np.int32)
        assert_tensor_equal(t, np.array(data, dtype=np.int32))

    def test_init_with_shape_and_dtype(self):
        shape = (2, 3)
        dtype = np.float32
        t = Tensor(shape=shape, dtype=dtype)
        assert t.shape == shape
        assert t.dtype == dtype
        assert t.device == "cpu"
        # Data should be uninitialized, so we can't compare values directly
        # but we can check if it's a numpy array of the correct shape and dtype
        assert isinstance(t.data, np.ndarray)
        assert t.data.shape == shape
        assert t.data.dtype == dtype

    def test_init_with_device(self):
        data = np.array([1.0], dtype=np.float32)
        t = Tensor(data=data, device="cpu")
        assert t.device == "cpu"

    def test_init_unsupported_data_type(self):
        with pytest.raises(TypeError, match="Unsupported data type"):
            Tensor(data="hello")

    def test_init_data_shape_mismatch(self):
        data = np.array([1, 2, 3])
        shape = (2, 2)
        with pytest.raises(ValueError, match="Provided data shape"):
            Tensor(data=data, shape=shape)

    def test_init_unsupported_numpy_dtype(self):
        # Use a numpy dtype that is not in _DTYPE_MAP
        data = np.array([1.0], dtype=np.complex64)
        with pytest.raises(ValueError, match="Unsupported numpy dtype"):
            Tensor(data=data)

    def test_init_no_data_or_shape(self):
        with pytest.raises(ValueError, match="Either 'data', 'cpp_node', or 'shape' must be provided"):
            Tensor()

    def test_init_with_cpp_node(self, sample_tensor):
        # This tests the internal mechanism, ensuring it can wrap a cpp_node
        # We create a tensor, then extract its internal cpp_node to create a new Tensor
        new_t = Tensor(cpp_node=sample_tensor._cpp_node)
        assert_tensor_equal(new_t, sample_tensor.data)
        assert new_t.shape == sample_tensor.shape
        assert new_t.dtype == sample_tensor.dtype
        assert new_t.device == sample_tensor.device


# Test Tensor properties
class TestTensorProperties:
    def test_shape_property(self, sample_tensor, sample_tensor_data):
        assert sample_tensor.shape == sample_tensor_data.shape

    def test_dtype_property(self, sample_tensor, sample_tensor_data):
        assert sample_tensor.dtype == sample_tensor_data.dtype.type

    def test_device_property(self, sample_tensor):
        assert sample_tensor.device == "cpu"

    def test_data_property(self, sample_tensor, sample_tensor_data):
        assert_tensor_equal(sample_tensor, sample_tensor_data)


# Test Unary Operations
class TestTensorUnaryOperations:
    def test_abs(self):
        data = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
        t = Tensor(data=data)
        result = t.abs()
        assert_tensor_equal(result, np.abs(data))

    def test_relu(self):
        data = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
        t = Tensor(data=data)
        result = t.relu()
        assert_tensor_equal(result, np.maximum(0, data))

    def test_leaky_relu(self):
        data = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
        t = Tensor(data=data)
        alpha = 0.1
        result = t.lrelu(alpha)
        expected = np.where(data > 0, data, data * alpha)
        assert_tensor_equal(result, expected)

    def test_exp(self):
        data = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        t = Tensor(data=data)
        result = t.exp()
        assert_tensor_equal(result, np.exp(data))

    def test_log(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t = Tensor(data=data)
        result = t.log()
        assert_tensor_equal(result, np.log(data))


# Test Binary Operations
class TestTensorBinaryOperations:
    def test_add_tensor_tensor(self, sample_tensor):
        other_data = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        other_t = Tensor(data=other_data)
        result = sample_tensor + other_t
        assert_tensor_equal(result, sample_tensor.data + other_data)

    def test_add_tensor_scalar(self, sample_tensor):
        scalar = 10.0
        result = sample_tensor + scalar
        assert_tensor_equal(result, sample_tensor.data + scalar)

    def test_radd_scalar_tensor(self, sample_tensor):
        scalar = 10.0
        result = scalar + sample_tensor
        assert_tensor_equal(result, scalar + sample_tensor.data)

    def test_sub_tensor_tensor(self, sample_tensor):
        other_data = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        other_t = Tensor(data=other_data)
        result = sample_tensor - other_t
        assert_tensor_equal(result, sample_tensor.data - other_data)

    def test_sub_tensor_scalar(self, sample_tensor):
        scalar = 10.0
        result = sample_tensor - scalar
        assert_tensor_equal(result, sample_tensor.data - scalar)

    def test_rsub_scalar_tensor(self, sample_tensor):
        scalar = 10.0
        result = scalar - sample_tensor
        assert_tensor_equal(result, scalar - sample_tensor.data)

    def test_mul_tensor_tensor(self, sample_tensor):
        other_data = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        other_t = Tensor(data=other_data)
        result = sample_tensor * other_t
        assert_tensor_equal(result, sample_tensor.data * other_data)

    def test_mul_tensor_scalar(self, sample_tensor):
        scalar = 10.0
        result = sample_tensor * scalar
        assert_tensor_equal(result, sample_tensor.data * scalar)

    def test_matmul_tensor_tensor(self):
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b_data = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

        # CPU Matmul
        cpu_a = Tensor(data=a_data, device="cpu")
        cpu_b = Tensor(data=b_data, device="cpu")

        start_cpu = time.perf_counter()
        cpu_result = cpu_a @ cpu_b
        end_cpu = time.perf_counter()
        cpu_time = end_cpu - start_cpu
        print(f"\nCPU Matmul (small test): {cpu_time:.6f} seconds")
        assert_tensor_equal(cpu_result, a_data @ b_data)

        # CUDA Matmul
        if Device.is_cuda_available():
            cuda_a = Tensor(data=a_data, device="cuda")
            cuda_b = Tensor(data=b_data, device="cuda")

            start_cuda = time.perf_counter()
            cuda_result = cuda_a @ cuda_b
            end_cuda = time.perf_counter()
            cuda_time = end_cuda - start_cuda
            print(f"CUDA Matmul (small test): {cuda_time:.6f} seconds")

            # Verify results are close
            assert_tensor_equal(cpu_result, cuda_result.data)
        else:
            print("CUDA not available, skipping CUDA matmul for small test")

    def test_add_broadcast(self):
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b_data = np.array([10.0, 20.0], dtype=np.float32) # Will broadcast to (2,2)
        a_t = Tensor(data=a_data)
        b_t = Tensor(data=b_data)
        result = a_t + b_t
        assert_tensor_equal(result, a_data + b_data)

    def test_mul_broadcast(self):
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b_data = np.array([10.0, 20.0], dtype=np.float32) # Will broadcast to (2,2)
        a_t = Tensor(data=a_data)
        b_t = Tensor(data=b_data)
        result = a_t * b_t
        assert_tensor_equal(result, a_data * b_data)


# Test Movement Operations
class TestTensorMovementOperations:
    def test_transpose_property(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t = Tensor(data=data)
        result = t.T
        assert_tensor_equal(result, data.T)

    def test_transpose_method(self):
        data = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)
        t = Tensor(data=data)
        result = t.transpose(0, 2) # Swap first and last dimension
        assert_tensor_equal(result, np.transpose(data, (2, 1, 0)))

    def test_reshape(self):
        data = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        t = Tensor(data=data)
        new_shape = (3, 2)
        result = t.reshape(new_shape)
        assert_tensor_equal(result, data.reshape(new_shape))

    def test_view(self):
        data = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        t = Tensor(data=data)
        result = t.view(3, 2)
        assert_tensor_equal(result, data.reshape(3, 2))

    def test_squeeze_all_dims(self):
        data = np.array([[[[1.0]]]], dtype=np.float32)
        t = Tensor(data=data)
        result = t.squeeze()
        assert_tensor_equal(result, data.squeeze())

    def test_squeeze_specific_dim(self):
        data = np.array([[[1.0, 2.0]]], dtype=np.float32)
        t = Tensor(data=data)
        result = t.squeeze(0)
        assert_tensor_equal(result, data.squeeze(0))

    def test_squeeze_no_op(self):
        data = np.array([[1.0, 2.0]], dtype=np.float32)
        t = Tensor(data=data)
        result = t.squeeze(0) # Dimension 0 has size 1, but it's not squeezed in numpy if it's the only dimension
        assert_tensor_equal(result, data.squeeze(0))

    def test_unsqueeze(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t = Tensor(data=data)
        result = t.unsqueeze(0)
        assert_tensor_equal(result, np.expand_dims(data, axis=0))

    def test_expand(self):
        data = np.array([1.0, 2.0], dtype=np.float32)
        t = Tensor(data=data)
        result = t.expand(2, 2)
        assert_tensor_equal(result, np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32))

    def test_broadcast_to(self):
        data = np.array([1.0, 2.0], dtype=np.float32)
        t = Tensor(data=data)
        target_shape = (2, 2)
        result = t.broadcast_to(*target_shape)
        assert_tensor_equal(result, np.broadcast_to(data, target_shape))

    def test_expand_with_new_dims(self):
        data = np.array([1.0, 2.0], dtype=np.float32)
        t = Tensor(data=data)
        result = t.expand(1, 2, 2)
        expected = np.array([[[1.0, 2.0], [1.0, 2.0]]], dtype=np.float32)
        assert_tensor_equal(result, expected)

    def test_expand_error_smaller_dims(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t = Tensor(data=data)
        with pytest.raises(ValueError, match="Cannot expand tensor from 2 dimensions to 1 dimensions"):
            t.expand(2)


# Test Reduction Operations
class TestTensorReductionOperations:
    def test_min_full(self, large_tensor, large_tensor_data):
        result = large_tensor.min()
        assert_tensor_equal(result, np.array(large_tensor_data.min(), dtype=np.float32))

    def test_min_dim(self, large_tensor, large_tensor_data):
        result = large_tensor.min(dim=1)
        assert_tensor_equal(result, large_tensor_data.min(axis=1))

    def test_min_dim_keepdim(self, large_tensor, large_tensor_data):
        result = large_tensor.min(dim=1, keepdim=True)
        assert_tensor_equal(result, np.expand_dims(large_tensor_data.min(axis=1), axis=1))

    def test_max_full(self, large_tensor, large_tensor_data):
        result = large_tensor.max()
        assert_tensor_equal(result, np.array(large_tensor_data.max(), dtype=np.float32))

    def test_max_dim(self, large_tensor, large_tensor_data):
        result = large_tensor.max(dim=1)
        assert_tensor_equal(result, large_tensor_data.max(axis=1))

    def test_max_dim_keepdim(self, large_tensor, large_tensor_data):
        result = large_tensor.max(dim=1, keepdim=True)
        assert_tensor_equal(result, np.expand_dims(large_tensor_data.max(axis=1), axis=1))

    def test_sum_full(self, large_tensor, large_tensor_data):
        result = large_tensor.sum()
        assert_tensor_equal(result, np.array(large_tensor_data.sum(), dtype=np.float32))

    def test_sum_dim(self, large_tensor, large_tensor_data):
        result = large_tensor.sum(dim=1)
        assert_tensor_equal(result, large_tensor_data.sum(axis=1))

    def test_sum_dim_keepdim(self, large_tensor, large_tensor_data):
        result = large_tensor.sum(dim=1, keepdim=True)
        assert_tensor_equal(result, np.expand_dims(large_tensor_data.sum(axis=1), axis=1))

    def test_mean_full(self, large_tensor, large_tensor_data):
        result = large_tensor.mean()
        assert_tensor_equal(result, np.array(large_tensor_data.mean(), dtype=np.float32))

    def test_mean_dim(self, large_tensor, large_tensor_data):
        result = large_tensor.mean(dim=1)
        assert_tensor_equal(result, large_tensor_data.mean(axis=1))

    def test_mean_dim_keepdim(self, large_tensor, large_tensor_data):
        result = large_tensor.mean(dim=1, keepdim=True)
        assert_tensor_equal(result, np.expand_dims(large_tensor_data.mean(axis=1), axis=1))


# Test Device Transfer
class TestTensorDeviceTransfer:
    def test_to_cpu(self, sample_tensor, sample_tensor_data):
        # Assuming default device is CPU, transferring to CPU should return a new tensor
        # with the same data and device.
        new_tensor = sample_tensor.to("cpu")
        assert new_tensor.device == "cpu"
        assert_tensor_equal(new_tensor, sample_tensor_data)
        # Ensure it's a new tensor, not the same object
        assert new_tensor is not sample_tensor

    @pytest.mark.skip(reason="CUDA device not available in current test environment")
    def test_to_cuda(self, sample_tensor):
        # This test would require a CUDA-enabled environment
        # For now, it's skipped.
        pass

    @pytest.mark.skipif(not Device.is_cuda_available(), reason="CUDA not available")
    def test_get_data_from_cuda_tensor(self, sample_tensor_data):
        # Create a tensor on CPU
        cpu_tensor = Tensor(data=sample_tensor_data, device="cpu")
        # Transfer it to CUDA
        cuda_tensor = cpu_tensor.to("cuda")
        assert cuda_tensor.device == "cuda"

        # Access the data property, which should trigger the copy from CUDA to CPU
        retrieved_data = cuda_tensor.data
        assert isinstance(retrieved_data, np.ndarray)
        np.testing.assert_allclose(retrieved_data, sample_tensor_data, rtol=1e-5, atol=1e-8)


# Test Utility Methods
class TestTensorUtilityMethods:
    def test_numel(self, sample_tensor, sample_tensor_data):
        assert sample_tensor.numel() == sample_tensor_data.size

    def test_numel_large_tensor(self, large_tensor, large_tensor_data):
        assert large_tensor.numel() == large_tensor_data.size


import time

class TestTensorBenchmarking:
    @pytest.mark.parametrize("size", [128, 256, 512])
    def test_matmul_performance(self, size):
        # Generate random matrices
        a_data = np.random.rand(size, size).astype(np.float32)
        b_data = np.random.rand(size, size).astype(np.float32)

        # CPU Matmul
        cpu_a = Tensor(data=a_data, device="cpu")
        cpu_b = Tensor(data=b_data, device="cpu")

        start_cpu = time.perf_counter()
        cpu_result = cpu_a @ cpu_b
        end_cpu = time.perf_counter()
        cpu_time = end_cpu - start_cpu
        print(f"\nCPU Matmul ({size}x{size}): {cpu_time:.6f} seconds")

        # CUDA Matmul
        if Device.is_cuda_available():
            cuda_a = Tensor(data=a_data, device="cuda")
            cuda_b = Tensor(data=b_data, device="cuda")

            start_cuda = time.perf_counter()
            cuda_result = cuda_a @ cuda_b
            end_cuda = time.perf_counter()
            cuda_time = end_cuda - start_cuda
            print(f"CUDA Matmul ({size}x{size}): {cuda_time:.6f} seconds")

            # Optional: Verify results are close
            assert_tensor_equal(cpu_result, cuda_result.data)
            assert cuda_time < cpu_time # Expect CUDA to be faster
        else:
            print(f"CUDA not available, skipping CUDA matmul benchmark for size {size}")