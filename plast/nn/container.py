from .module import Module


class Sequential(Module):
    """A sequential container of modules.

    Modules are applied in the order they are added.  You can pass them as
    positional arguments or as a ``dict`` with named keys::

        # positional — modules named '0', '1', '2', …
        model = plast.nn.Sequential(
            plast.nn.Linear(784, 256),
            plast.nn.ReLU(),
            plast.nn.Linear(256, 10),
        )

        # named
        model = plast.nn.Sequential({
            "embed": plast.nn.Linear(784, 256),
            "act":   plast.nn.ReLU(),
            "head":  plast.nn.Linear(256, 10),
        })

        out = model(x)
    """

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], dict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def add_module(self, name: str, module: Module) -> None:
        """Append *module* under *name*."""
        setattr(self, name, module)
        self._modules[name] = module

    def append(self, module: Module) -> "Sequential":
        """Append *module* and return *self* (chainable)."""
        self.add_module(str(len(self._modules)), module)
        return self

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

    def __getitem__(self, idx):
        modules = list(self._modules.values())
        if isinstance(idx, slice):
            return Sequential(*modules[idx])
        return modules[idx]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __repr__(self) -> str:
        lines = ["Sequential("]
        for name, module in self._modules.items():
            mod_repr = repr(module)
            mod_lines = mod_repr.splitlines()
            indented = "\n".join("  " + l for l in mod_lines)
            lines.append(f"  ({name}): {indented.lstrip()}")
        lines.append(")")
        return "\n".join(lines)


class ModuleList(Module):
    """Holds sub-modules in a list.

    :class:`ModuleList` is registered just like a regular list, but each
    element is a proper sub-module (its parameters are discoverable via
    :meth:`~Module.parameters`)::

        layers = plast.nn.ModuleList([
            plast.nn.Linear(128, 128)
            for _ in range(6)
        ])
        for layer in layers:
            x = layer(x)
    """

    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            for idx, module in enumerate(modules):
                self.add_module(str(idx), module)

    def add_module(self, name: str, module: Module) -> None:
        setattr(self, name, module)
        self._modules[name] = module

    def append(self, module: Module) -> "ModuleList":
        """Append *module* and return *self* (chainable)."""
        self.add_module(str(len(self._modules)), module)
        return self

    def __getitem__(self, idx):
        modules = list(self._modules.values())
        if isinstance(idx, slice):
            return ModuleList(modules[idx])
        if idx < 0:
            idx += len(modules)
        if not (0 <= idx < len(modules)):
            raise IndexError(
                f"ModuleList index {idx} is out of range (size {len(modules)})."
            )
        return modules[idx]

    def __setitem__(self, idx, module):
        keys = list(self._modules.keys())
        key = keys[idx]
        setattr(self, key, module)
        self._modules[key] = module

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def forward(self, *args, **kwargs):
        raise TypeError(
            "ModuleList is not a callable module. "
            "Iterate over it and call each sub-module individually."
        )

    def __repr__(self) -> str:
        lines = ["ModuleList("]
        for name, module in self._modules.items():
            mod_repr = repr(module)
            mod_lines = mod_repr.splitlines()
            indented = "\n".join("  " + l for l in mod_lines)
            lines.append(f"  ({name}): {indented.lstrip()}")
        lines.append(")")
        return "\n".join(lines)
