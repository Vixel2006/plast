from ..plast_core import Device, zero_grad_cpu

try:
    from ..plast_core import zero_grad_cuda
except ImportError:
    zero_grad_cuda = None


class Optimizer:
    """Base class for all optimizers.

    Do **not** instantiate this directly; use one of the concrete
    sub-classes (:class:`~plast.optim.SGD`, :class:`~plast.optim.Adam`,
    :class:`~plast.optim.AdamW`).

    Parameters are tracked as *param groups*, allowing different
    hyperparameters for different subsets::

        opt = plast.optim.Adam([
            {"params": backbone.parameters(), "lr": 1e-4},
            {"params": head.parameters(),     "lr": 1e-3},
        ])
    """

    def __init__(self, params, defaults: dict):
        if isinstance(params, (list, tuple)) and params and isinstance(params[0], dict):
            # param groups were passed directly
            self.param_groups = []
            for group in params:
                merged = {**defaults, **group}
                self.param_groups.append(merged)
            self.params = [p for g in self.param_groups for p in g["params"]]
        else:
            self.params = list(params)
            if not self.params:
                raise ValueError(
                    "Optimizer received an empty parameter list. "
                    "Did you forget to call model.parameters()?"
                )
            self.param_groups = [{"params": self.params, **defaults}]
        self.defaults = defaults

    @property
    def lr(self) -> float:
        """Current learning rate of the first param group."""
        return self.param_groups[0]["lr"]

    @lr.setter
    def lr(self, value: float) -> None:
        """Set the learning rate for **all** param groups."""
        for g in self.param_groups:
            g["lr"] = value

    def step(self) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} must implement a step() method."
        )

    def zero_grad(self) -> None:
        """Zero out the gradients of all tracked parameters."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    raw = p._t
                    if p.device == Device.CUDA:
                        if zero_grad_cuda is not None:
                            zero_grad_cuda(raw)
                    else:
                        zero_grad_cpu(raw)

    def state_dict(self) -> dict:
        """Return optimizer state (hyperparameters) as a plain dict."""
        return {
            "param_groups": [
                {k: v for k, v in g.items() if k != "params"}
                for g in self.param_groups
            ]
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore optimizer hyperparameters from *state_dict*."""
        groups = state_dict.get("param_groups", [])
        for i, (group, saved) in enumerate(zip(self.param_groups, groups)):
            group.update(saved)

    def __repr__(self) -> str:
        lrs = set(g["lr"] for g in self.param_groups)
        lr_str = next(iter(lrs)) if len(lrs) == 1 else f"[{', '.join(str(l) for l in lrs)}]"
        return f"{type(self).__name__}(lr={lr_str})"
