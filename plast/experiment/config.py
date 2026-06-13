import os
import logging
from .yaml_utils import load_yaml, dump_yaml

logger = logging.getLogger(__name__)


class ExperimentConfig:
    """Configuration object for an experiment run.

    Args:
        name:     Experiment name (used as the top-level directory).
        model:    Dict of model hyperparameters (architecture, size, etc.).
        training: Dict of training hyperparameters (lr, epochs, batch_size, …).
        device:   Device string — ``'cpu'`` or ``'cuda'`` (default ``'cpu'``).
        seed:     Random seed for reproducibility (default ``42``).
        notes:    Free-form notes about this experiment.

    Example::

        config = ExperimentConfig(
            name="mlp_mnist",
            model={"hidden": 256, "dropout": 0.2},
            training={"lr": 1e-3, "epochs": 50, "batch_size": 64},
            device="cuda",
            seed=0,
        )
    """

    def __init__(
        self,
        name: str,
        model: dict,
        training: dict,
        device: str = "cpu",
        seed: int = 42,
        notes: str = "",
    ):
        self.name = name
        self.model = model
        self.training = training
        self.device = device
        self.seed = seed
        self.notes = notes

    def to_dict(self) -> dict:
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

    def save(self, filepath: str) -> None:
        """Serialize the config to a YAML file at *filepath*."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(dump_yaml(self.to_dict()))
        logger.debug("Config saved to %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> "ExperimentConfig":
        """Load an :class:`ExperimentConfig` from a YAML file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"ExperimentConfig.load: file not found: '{filepath}'"
            )
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

    def __repr__(self) -> str:
        return (
            f"ExperimentConfig(name='{self.name}', device='{self.device}', "
            f"seed={self.seed})"
        )
