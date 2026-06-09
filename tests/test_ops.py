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


class TestConv2d:
    def _conv2d_ref(self, x, w, stride=1):
        N, C, H, W = x.shape
        F, _, KH, KW = w.shape
        H_out = (H - KH) // stride + 1
        W_out = (W - KW) // stride + 1
        out = np.zeros((N, F, H_out, W_out), dtype=np.float32)
        for n in range(N):
            for f in range(F):
                for h in range(H_out):
                    for wi in range(W_out):
                        hs = h * stride
                        ws = wi * stride
                        out[n, f, h, wi] = np.sum(
                            x[n, :, hs : hs + KH, ws : ws + KW] * w[f, :, :, :]
                        )
        return out

    @pytest.mark.parametrize("N,C,H,W,F,KH,KW,stride", [
        (2, 1, 8, 8, 2, 3, 3, 1),
        (1, 3, 8, 8, 4, 1, 1, 1),
        (1, 3, 8, 8, 4, 5, 5, 1),
        (1, 3, 8, 8, 4, 3, 3, 2),
        (2, 2, 10, 10, 3, 3, 3, 2),
    ])
    def test_conv2d_forward(self, N, C, H, W, F, KH, KW, stride, device, tol, rng):
        x_np = rng.randn(N, C, H, W).astype(np.float32)
        w_np = rng.randn(F, C, KH, KW).astype(np.float32)
        x = plast.tensor(x_np, device=device)
        w = plast.tensor(w_np, device=device)
        out = plast.nn.functional.conv2d(x, w, stride=stride)
        plast.forward(out)
        expected = self._conv2d_ref(x_np, w_np, stride=stride)
        np.testing.assert_allclose(out.numpy(), expected, **tol)

    @pytest.mark.parametrize("N,C,H,W,F,KH,KW,stride", [
        (1, 2, 8, 8, 3, 3, 3, 1),
        (1, 1, 6, 6, 2, 3, 3, 2),
    ])
    def test_conv2d_forward_with_bias(self, N, C, H, W, F, KH, KW, stride, device, tol, rng):
        x_np = rng.randn(N, C, H, W).astype(np.float32)
        w_np = rng.randn(F, C, KH, KW).astype(np.float32)
        bias_np = rng.randn(F).astype(np.float32)
        x = plast.tensor(x_np, device=device)
        w = plast.tensor(w_np, device=device)
        bias = plast.tensor(bias_np, device=device)
        out = plast.nn.functional.conv2d(x, w, bias, stride=stride)
        plast.forward(out)
        expected = self._conv2d_ref(x_np, w_np, stride=stride) + bias_np.reshape(1, F, 1, 1)
        np.testing.assert_allclose(out.numpy(), expected, **tol)

    @pytest.mark.parametrize("N,C,H,W,F,KH,KW,stride", [
        (1, 2, 5, 5, 2, 3, 3, 1),
    ])
    def test_conv2d_backward(self, N, C, H, W, F, KH, KW, stride, device, tol, rng):
        H_out = (H - KH) // stride + 1
        W_out = (W - KW) // stride + 1

        x_np = rng.randn(N, C, H, W).astype(np.float32)
        w_np = rng.randn(F, C, KH, KW).astype(np.float32)
        x = plast.tensor(x_np, device=device, requires_grad=True)
        w = plast.tensor(w_np, device=device, requires_grad=True)
        out = plast.nn.functional.conv2d(x, w, stride=stride)
        loss = out.sum()
        plast.forward(loss)
        loss.backward()

        assert x.grad is not None
        assert w.grad is not None

        # dL/dX[n,c,h,w] = sum_{f,kh,kw} W[f,c,kh,kw]
        #   for all (kh,kw) where h-oh*stride=kh, w-ow*stride=kw
        expected_x_grad = np.zeros_like(x_np)
        for n in range(N):
            for c in range(C):
                for h in range(H):
                    for wi in range(W):
                        val = 0.0
                        for f in range(F):
                            for kh_i in range(KH):
                                for kw_i in range(KW):
                                    oh = h - kh_i
                                    ow = wi - kw_i
                                    if oh % stride == 0 and ow % stride == 0:
                                        oh //= stride
                                        ow //= stride
                                        if 0 <= oh < H_out and 0 <= ow < W_out:
                                            val += w_np[f, c, kh_i, kw_i]
                        expected_x_grad[n, c, h, wi] = val

        # dL/dW[f,c,kh,kw] = sum_{n,oh,ow} X[n,c,oh*stride+kh,ow*stride+kw]
        expected_w_grad = np.zeros_like(w_np)
        for f in range(F):
            for c in range(C):
                for kh_i in range(KH):
                    for kw_i in range(KW):
                        val = 0.0
                        for n in range(N):
                            for oh in range(H_out):
                                for ow in range(W_out):
                                    h = oh * stride + kh_i
                                    wi = ow * stride + kw_i
                                    val += x_np[n, c, h, wi]
                        expected_w_grad[f, c, kh_i, kw_i] = val

        np.testing.assert_allclose(x.grad.numpy(), expected_x_grad, **tol)
        np.testing.assert_allclose(w.grad.numpy(), expected_w_grad, **tol)


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
