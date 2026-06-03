import os
from .yaml_utils import load_yaml, dump_yaml

class ExperimentConfig:
    def __init__(self, name, model, training, device="cpu", seed=42, notes=""):
        self.name = name
        self.model = model
        self.training = training
        self.device = device
        self.seed = seed
        self.notes = notes

    def to_dict(self):
        return {
            "experiment": {
                "name": self.name,
                "device": self.device,
                "seed": self.seed,
                "notes": self.notes,
            },
            "model": self.model,
            "training": self.training,
        }

    def save(self, filepath):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(dump_yaml(self.to_dict()))

        # For debug and pretty view, print where we saved it
        print(f"Config saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        with open(filepath, "r") as f:
            data = load_yaml(f.read())
        exp = data.get("experiment", {})
        return cls(
            name=exp.get("name", "experiment"),
            model=data.get("model", {}),
            training=data.get("training", {}),
            device=exp.get("device", "cpu"),
            seed=exp.get("seed", 42),
            notes=exp.get("notes", ""),
        )
