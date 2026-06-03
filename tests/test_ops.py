import numpy as np
import pytest
import plast

SHAPES = [
    (4,),
    (2, 3),
    (3, 2, 4),
]

BROADCAST_PAIRS = [
    ((4,), (1,)),
    ((2, 3), (1, 3)),
    ((3, 1, 4), (1, 2, 1)),
]


class TestElementWiseOps:
    @pytest.mark.parametrize("shape", SHAPES)
    def test_add_forward(self, shape, device, tol, rng):
        a = rng.randn(*shape).astype(np.float32)
        b = rng.randn(*shape).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tb = plast.tensor(b, device=device)
        tc = ta + tb
        plast.forward(tc)
        expected = a + b
        np.testing.assert_allclose(tc.numpy(), expected, **tol)

    @pytest.mark.parametrize("shape_a, shape_b", BROADCAST_PAIRS)
    def test_add_broadcast(self, shape_a, shape_b, device, tol, rng):
        a = rng.randn(*shape_a).astype(np.float32)
        b = rng.randn(*shape_b).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tb = plast.tensor(b, device=device)
        tc = ta + tb
        plast.forward(tc)
        expected = a + b
        np.testing.assert_allclose(tc.numpy(), expected, **tol)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_sub_forward(self, shape, device, tol, rng):
        a, b = [rng.randn(*shape).astype(np.float32) for _ in range(2)]
        ta, tb = [plast.tensor(x, device=device) for x in (a, b)]
        tc = ta - tb
        plast.forward(tc)
        np.testing.assert_allclose(tc.numpy(), a - b, **tol)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_mul_forward(self, shape, device, tol, rng):
        a, b = [rng.randn(*shape).astype(np.float32) for _ in range(2)]
        ta, tb = [plast.tensor(x, device=device) for x in (a, b)]
        tc = ta * tb
        plast.forward(tc)
        np.testing.assert_allclose(tc.numpy(), a * b, **tol)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_div_forward(self, shape, device, tol, rng):
        a = rng.randn(*shape).astype(np.float32)
        b = np.abs(rng.randn(*shape)).astype(np.float32) + 0.1
        ta, tb = [plast.tensor(x, device=device) for x in (a, b)]
        tc = ta / tb
        plast.forward(tc)
        np.testing.assert_allclose(tc.numpy(), a / b, **tol)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_neg_forward(self, shape, device, tol, rng):
        a = rng.randn(*shape).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tc = -ta
        plast.forward(tc)
        np.testing.assert_allclose(tc.numpy(), -a, **tol)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_abs_forward(self, shape, device, tol, rng):
        a = rng.randn(*shape).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tc = abs(ta)
        plast.forward(tc)
        np.testing.assert_allclose(tc.numpy(), np.abs(a), **tol)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_exp_forward(self, shape, device, tol, rng):
        a = rng.randn(*shape).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tc = ta.exp()
        plast.forward(tc)
        np.testing.assert_allclose(tc.numpy(), np.exp(a), atol=1e-3)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_log_forward(self, shape, device, tol, rng):
        a = np.abs(rng.randn(*shape)).astype(np.float32) + 0.1
        ta = plast.tensor(a, device=device)
        tc = ta.log()
        plast.forward(tc)
        np.testing.assert_allclose(tc.numpy(), np.log(a), **tol)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_sin_forward(self, shape, device, tol, rng):
        a = rng.randn(*shape).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tc = ta.sin()
        plast.forward(tc)
        np.testing.assert_allclose(tc.numpy(), np.sin(a), **tol)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_cos_forward(self, shape, device, tol, rng):
        a = rng.randn(*shape).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tc = ta.cos()
        plast.forward(tc)
        np.testing.assert_allclose(tc.numpy(), np.cos(a), **tol)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_tan_forward(self, shape, device, tol, rng):
        a = rng.randn(*shape).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tc = ta.tan()
        plast.forward(tc)
        np.testing.assert_allclose(tc.numpy(), np.tan(a), **tol)


class TestReductionOps:
    @pytest.mark.parametrize("shape", SHAPES)
    def test_sum_forward(self, shape, device, tol, rng):
        a = rng.randn(*shape).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tc = ta.sum()
        plast.forward(tc)
        expected = np.sum(a)
        np.testing.assert_allclose(tc.numpy(), np.array([expected]), **tol)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_sum_dim(self, shape, device, tol, rng):
        a = rng.randn(*shape).astype(np.float32)
        ta = plast.tensor(a, device=device)
        for dim in range(len(shape)):
            tc = ta.sum(dim=dim)
            plast.forward(tc)
            expected = np.sum(a, axis=dim)
            np.testing.assert_allclose(tc.numpy(), expected, **tol)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_mean_forward(self, shape, device, tol, rng):
        a = rng.randn(*shape).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tc = ta.mean()
        plast.forward(tc)
        expected = np.mean(a)
        np.testing.assert_allclose(tc.numpy(), np.array([expected]), **tol)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_mean_dim(self, shape, device, tol, rng):
        a = rng.randn(*shape).astype(np.float32)
        ta = plast.tensor(a, device=device)
        for dim in range(len(shape)):
            tc = ta.mean(dim=dim)
            plast.forward(tc)
            expected = np.mean(a, axis=dim)
            np.testing.assert_allclose(tc.numpy(), expected, **tol)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_max_forward(self, shape, device, tol, rng):
        a = rng.randn(*shape).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tc = ta.max()
        plast.forward(tc)
        expected = np.max(a)
        np.testing.assert_allclose(tc.numpy(), np.array([expected]), **tol)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_min_forward(self, shape, device, tol, rng):
        a = rng.randn(*shape).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tc = ta.min()
        plast.forward(tc)
        expected = np.min(a)
        np.testing.assert_allclose(tc.numpy(), np.array([expected]), **tol)

    @pytest.mark.xfail(reason="max with dim returns wrong output")
    @pytest.mark.parametrize("shape", [(2, 3), (3, 4)])
    def test_max_dim(self, shape, device, tol, rng):
        a = rng.randn(*shape).astype(np.float32)
        ta = plast.tensor(a, device=device)
        for dim in range(len(shape)):
            tc = ta.max(dim=dim)
            plast.forward(tc)
            expected = np.max(a, axis=dim, keepdims=True)
            np.testing.assert_allclose(tc.numpy(), expected, **tol)


class TestMatmul:
    @pytest.mark.parametrize("m,n,p", [(2, 3, 4), (4, 8, 2), (1, 5, 1)])
    def test_matmul_forward(self, m, n, p, device, tol, rng):
        a = rng.randn(m, n).astype(np.float32)
        b = rng.randn(n, p).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tb = plast.tensor(b, device=device)
        tc = ta @ tb
        plast.forward(tc)
        expected = a @ b
        np.testing.assert_allclose(tc.numpy(), expected, **tol)

    def test_matmul_shape_mismatch(self, device, rng):
        a = plast.tensor(rng.randn(2, 3).astype(np.float32), device=device)
        b = plast.tensor(rng.randn(4, 5).astype(np.float32), device=device)
        with pytest.raises((ValueError, RuntimeError)):
            _ = a @ b
            plast.forward(_)


class TestShapeOps:
    @pytest.mark.xfail(reason="view + element-wise op produces incorrect non-contiguous strides")
    @pytest.mark.parametrize("shape", [(2, 3, 4), (4, 6)])
    def test_view(self, shape, device, tol, rng):
        total = int(np.prod(shape))
        a = rng.randn(*shape).astype(np.float32)
        ta = plast.tensor(a, device=device)
        new_shape = [shape[0], -1]
        tc = ta.view(*new_shape)
        plast.forward(tc)
        expected = a.reshape(shape[0], -1)
        np.testing.assert_allclose(tc.numpy(), expected, **tol)

    @pytest.mark.xfail(reason="view with -1 produces incorrect output")
    def test_view_negative_one(self, device, tol, rng):
        a = rng.randn(2, 3, 4).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tc = ta.view(2, -1)
        plast.forward(tc)
        np.testing.assert_allclose(tc.numpy(), a.reshape(2, -1), **tol)

    @pytest.mark.xfail(reason="reshape (which calls view) produces incorrect output")
    def test_reshape(self, device, tol, rng):
        a = rng.randn(2, 6).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tc = ta.reshape(3, 4)
        plast.forward(tc)
        np.testing.assert_allclose(tc.numpy(), a.reshape(3, 4), **tol)

    @pytest.mark.xfail(reason="transpose produces incorrect output for 2D+ tensors")
    @pytest.mark.parametrize("shape", [(2, 3, 4), (4, 5)])
    def test_transpose(self, shape, device, tol, rng):
        a = rng.randn(*shape).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tc = ta.transpose(0, 1)
        plast.forward(tc)
        expected = np.transpose(a, (1, 0, 2)) if len(shape) == 3 else a.T
        np.testing.assert_allclose(tc.numpy(), expected, **tol)

    @pytest.mark.xfail(reason="squeeze with no dim doesn't remove all squeeze dims")
    def test_squeeze(self, device, tol, rng):
        a = rng.randn(1, 3, 1, 4).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tc = ta.squeeze()
        plast.forward(tc)
        assert tc.shape == [3, 4]

    def test_unsqueeze(self, device, tol, rng):
        a = rng.randn(3, 4).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tc = ta.unsqueeze(0)
        plast.forward(tc)
        assert tc.shape == [1, 3, 4]
        np.testing.assert_allclose(tc.numpy(), a[np.newaxis, ...], **tol)

    def test_flatten(self, device, tol, rng):
        a = rng.randn(2, 3, 4).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tc = ta.flatten(1, 2)
        plast.forward(tc)
        expected = a.reshape(2, -1)
        np.testing.assert_allclose(tc.numpy(), expected, **tol)

    @pytest.mark.xfail(reason="expand produces incorrect output")
    def test_expand(self, device, tol, rng):
        a = rng.randn(1, 3, 1).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tc = ta.expand(4, 3, 5)
        plast.forward(tc)
        expected = np.broadcast_to(a, (4, 3, 5))
        np.testing.assert_allclose(tc.numpy(), expected, **tol)


class TestLeakyReLU:
    @pytest.mark.xfail(reason="leaky_relu forward does not apply alpha factor")
    @pytest.mark.parametrize("shape", SHAPES)
    def test_leaky_relu_forward(self, shape, device, tol, rng):
        a = rng.randn(*shape).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tc = plast.nn.functional.leaky_relu(ta, 0.01)
        plast.forward(tc)
        expected = np.where(a > 0, a, 0.01 * a)
        np.testing.assert_allclose(tc.numpy(), expected, **tol)


class TestOpsWithGradients:
    @pytest.mark.parametrize("shape", [(2, 3)])
    def test_add_backward(self, shape, device, rng):
        a = rng.randn(*shape).astype(np.float32)
        b = rng.randn(*shape).astype(np.float32)
        ta = plast.tensor(a, device=device, requires_grad=True)
        tb = plast.tensor(b, device=device, requires_grad=True)
        tc = ta + tb
        loss = tc.sum()
        plast.forward(loss)
        loss.backward()
        assert ta.grad is not None
        np.testing.assert_allclose(ta.grad.numpy(), np.ones(shape))

    @pytest.mark.parametrize("shape", [(2, 3)])
    def test_mul_backward(self, shape, device, tol, rng):
        a = rng.randn(*shape).astype(np.float32)
        b = rng.randn(*shape).astype(np.float32)
        ta = plast.tensor(a, device=device, requires_grad=True)
        tb = plast.tensor(b, device=device, requires_grad=True)
        tc = ta * tb
        loss = tc.sum()
        plast.forward(loss)
        loss.backward()
        np.testing.assert_allclose(ta.grad.numpy(), b, **tol)
        np.testing.assert_allclose(tb.grad.numpy(), a, **tol)

    @pytest.mark.xfail(reason="matmul backward produces incorrect gradients")
    @pytest.mark.parametrize("m,n,p", [(2, 3, 4)])
    def test_matmul_backward(self, m, n, p, device, tol, rng):
        a = rng.randn(m, n).astype(np.float32)
        b = rng.randn(n, p).astype(np.float32)
        ta = plast.tensor(a, device=device, requires_grad=True)
        tb = plast.tensor(b, device=device, requires_grad=True)
        tc = ta @ tb
        loss = tc.sum()
        plast.forward(loss)
        loss.backward()
        np.testing.assert_allclose(ta.grad.numpy(), np.ones((m, n)) @ b.T, **tol)
        np.testing.assert_allclose(tb.grad.numpy(), a.T @ np.ones((n, p)), **tol)
