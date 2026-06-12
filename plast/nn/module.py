import numpy as np
from ..tensor import Tensor
from ..plast_core import Device, zero_grad_cpu

try:
    from ..plast_core import zero_grad_cuda
except ImportError:
    zero_grad_cuda = None


class Module:
    """Base class for all neural network modules.

    Your model should subclass :class:`Module` and implement
    :meth:`forward`::

        class MLP(plast.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = plast.nn.Linear(784, 256)
                self.fc2 = plast.nn.Linear(256, 10)

            def forward(self, x):
                x = plast.nn.functional.relu(self.fc1(x))
                return self.fc2(x)

    Sub-modules assigned as attributes are automatically registered,
    as are :class:`~plast.Parameter` values.
    """

    def __init__(self):
        self.training = True
        self._modules: dict = {}
        self._parameters: dict = {}

    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            # Remove from _modules if previously registered there
            if hasattr(self, "_modules") and name in self._modules:
                del self._modules[name]
            if hasattr(self, "_parameters"):
                self._parameters[name] = value
        elif isinstance(value, Module):
            # Remove from _parameters if previously registered there
            if hasattr(self, "_parameters") and name in self._parameters:
                del self._parameters[name]
            if hasattr(self, "_modules"):
                self._modules[name] = value
        else:
            # For non-Tensor/Module assignments, clear from registries if needed
            if hasattr(self, "_parameters") and name in self._parameters:
                del self._parameters[name]
            if hasattr(self, "_modules") and name in self._modules:
                del self._modules[name]
        super().__setattr__(name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement a forward() method."
        )

    # ── Parameter access ──

    def parameters(self):
        """Return a list of all learnable :class:`~plast.Parameter` tensors."""
        params = list(self._parameters.values())
        for m in self._modules.values():
            params.extend(m.parameters())
        return params

    def named_parameters(self, prefix=""):
        """Return an iterator of ``(name, param)`` pairs."""
        named_params = []
        for name, param in self._parameters.items():
            named_params.append((f"{prefix}{name}", param))
        for m_name, m in self._modules.items():
            named_params.extend(m.named_parameters(f"{prefix}{m_name}."))
        return named_params

    def num_parameters(self, only_trainable: bool = True) -> int:
        """Return the total number of scalar parameters.

        Args:
            only_trainable: If ``True`` (default), count only parameters
                            with ``requires_grad=True``.

        Example::

            print(f"Parameters: {model.num_parameters():,}")
        """
        total = 0
        for p in self.parameters():
            if only_trainable and not p.requires_grad:
                continue
            total += p.numel()
        return total

    def named_modules(self, prefix="", memo=None):
        """Return an iterator of ``(name, module)`` pairs for all sub-modules."""
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            yield prefix, self
            for name, m in self._modules.items():
                full_name = f"{prefix}.{name}" if prefix else name
                yield from m.named_modules(prefix=full_name, memo=memo)

    def modules(self):
        """Return an iterator over all sub-modules (including self)."""
        for _, m in self.named_modules():
            yield m

    # ── Training / eval mode ──

    def train(self, mode: bool = True) -> "Module":
        """Set the module and all sub-modules to training mode."""
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> "Module":
        """Set the module and all sub-modules to eval mode."""
        return self.train(False)

    # ── Device transfer ──

    def to(self, device) -> "Module":
        """Move all parameters to *device* and return *self*.

        Example::

            model.to(plast.Device.CUDA)
        """
        for name, param in list(self._parameters.items()):
            moved = param.to(device)
            # Replace the registered parameter in-place
            self._parameters[name] = moved
            super(Module, self).__setattr__(name, moved)
        for m in self._modules.values():
            m.to(device)
        return self

    def cuda(self) -> "Module":
        """Move all parameters to CUDA."""
        return self.to(Device.CUDA)

    def cpu(self) -> "Module":
        """Move all parameters to CPU."""
        return self.to(Device.CPU)

    # ── Gradient management ──

    def zero_grad(self) -> None:
        """Zero out gradients of all parameters."""
        for p in self.parameters():
            raw = p._t
            if p.device == Device.CUDA:
                if zero_grad_cuda is not None:
                    zero_grad_cuda(raw)
            else:
                zero_grad_cpu(raw)

    # ── State dict (checkpointing) ──

    def state_dict(self, prefix="") -> dict:
        """Return a flat ``dict`` mapping parameter names to numpy arrays."""
        state = {}
        for name, param in self._parameters.items():
            state[f"{prefix}{name}"] = param.numpy()
        for m_name, m in self._modules.items():
            state.update(m.state_dict(f"{prefix}{m_name}."))
        return state

    def load_state_dict(self, state_dict: dict, prefix="", strict: bool = True) -> None:
        """Load parameters from *state_dict*.

        Args:
            state_dict: Dict mapping parameter names to numpy arrays.
            prefix:     Internal prefix used for nested modules (usually left empty).
            strict:     If ``True``, raises an error for missing keys.
        """
        for name, param in self._parameters.items():
            key = f"{prefix}{name}"
            if key in state_dict:
                arr = np.asarray(state_dict[key], dtype=np.float32)
                if list(arr.shape) != param.shape:
                    raise ValueError(
                        f"load_state_dict: shape mismatch for '{key}': "
                        f"model has {param.shape}, checkpoint has {list(arr.shape)}."
                    )
                param.copy_from_numpy(arr)
            elif strict:
                raise KeyError(
                    f"load_state_dict: key '{key}' is missing from the checkpoint. "
                    "Pass strict=False to ignore missing keys."
                )
        for m_name, m in self._modules.items():
            m.load_state_dict(state_dict, f"{prefix}{m_name}.", strict=strict)

    # ── Repr ──

    def extra_repr(self) -> str:
        """Override to add extra info to :meth:`__repr__`."""
        return ""

    def __repr__(self) -> str:
        extra = self.extra_repr()
        if not self._modules:
            return f"{type(self).__name__}({extra})"
        lines = [f"{type(self).__name__}("]
        if extra:
            lines[0] = f"{type(self).__name__}({extra},"
        for name, module in self._modules.items():
            mod_repr = repr(module)
            mod_lines = mod_repr.splitlines()
            indented = "\n".join("  " + l for l in mod_lines)
            lines.append(f"  ({name}): {indented.lstrip()}")
        lines.append(")")
        return "\n".join(lines)
