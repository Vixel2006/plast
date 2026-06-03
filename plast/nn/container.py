from .module import Module


class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], dict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def add_module(self, name, module):
        setattr(self, name, module)
        self._modules[name] = module

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

    def __repr__(self):
        lines = []
        for name, module in self._modules.items():
            mod_str = repr(module)
            mod_str = "\n".join(["  " + line for line in mod_str.split("\n")])
            lines.append(f"  ({name}): {mod_str}")
        lines_str = "\n".join(lines)
        return f"Sequential(\n{lines_str}\n)"


class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            for idx, module in enumerate(modules):
                self.add_module(str(idx), module)

    def add_module(self, name, module):
        setattr(self, name, module)
        self._modules[name] = module

    def __getitem__(self, idx):
        return list(self._modules.values())[idx]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __repr__(self):
        lines = []
        for name, module in self._modules.items():
            mod_str = repr(module)
            mod_str = "\n".join(["  " + line for line in mod_str.split("\n")])
            lines.append(f"  ({name}): {mod_str}")
        lines_str = "\n".join(lines)
        return f"ModuleList(\n{lines_str}\n)"
