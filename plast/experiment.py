import os
import yaml
from typing import Any, Dict, List, Optional
import numpy as np

class Experiment:
    def __init__(self, id: str, name: str, description: str = ""):
        self.id = id
        self.name = name
        self.description = description
        self.hyperparameters: Dict[str, Any] = {}
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.artifacts: List[Dict[str, str]] = []

    def log_hyperparameter(self, key: str, value: Any):
        self.hyperparameters[key] = value

    def log_metric(self, key: str, value: Any, step: Optional[int] = None):
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append({"value": value, "step": step})

    def log_artifact(self, path: str, name: Optional[str] = None):
        self.artifacts.append({"path": path, "name": name if name else os.path.basename(path)})

    def save(self, directory: str = "experiments"):
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f"{self.id}.yaml")
        data = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }
        with open(file_path, "w") as f:
            yaml.dump(self._convert_numpy_to_native(data), f, default_flow_style=False)
        print(f"Experiment '{self.id}' saved to {file_path}")

    def _convert_numpy_to_native(self, obj):
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_native(elem) for elem in obj]
        return obj

    @staticmethod
    def load(experiment_id: str, directory: str = "experiments") -> 'Experiment':
        file_path = os.path.join(directory, f"{experiment_id}.yaml")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Experiment file not found: {file_path}")
        
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        
        exp = Experiment(id=data["id"], name=data["name"], description=data.get("description", ""))
        exp.hyperparameters = data.get("hyperparameters", {})
        exp.metrics = data.get("metrics", {})
        exp.artifacts = data.get("artifacts", [])
        return exp

