import os

import pytest
import yaml

from plast.experiment import Experiment


class TestExperiment:
    def setup_method(self):
        self.test_dir = "test_experiments"
        os.makedirs(self.test_dir, exist_ok=True)

    def teardown_method(self):
        if os.path.exists(self.test_dir):
            for f in os.listdir(self.test_dir):
                os.remove(os.path.join(self.test_dir, f))
            os.rmdir(self.test_dir)

    def test_experiment_init(self):
        exp = Experiment(
            id="exp1", name="My First Experiment", description="A test run"
        )
        assert exp.id == "exp1"
        assert exp.name == "My First Experiment"
        assert exp.description == "A test run"
        assert exp.hyperparameters == {}
        assert exp.metrics == {}
        assert exp.artifacts == []

    def test_log_hyperparameter(self):
        exp = Experiment(id="exp2", name="Hyperparam Test")
        exp.log_hyperparameter("learning_rate", 0.01)
        exp.log_hyperparameter("batch_size", 32)
        assert exp.hyperparameters == {"learning_rate": 0.01, "batch_size": 32}

    def test_log_metric(self):
        exp = Experiment(id="exp3", name="Metric Test")
        exp.log_metric("accuracy", 0.95, step=10)
        exp.log_metric("loss", 0.1, step=10)
        exp.log_metric("accuracy", 0.96, step=20)
        assert exp.metrics == {
            "accuracy": [{"value": 0.95, "step": 10}, {"value": 0.96, "step": 20}],
            "loss": [{"value": 0.1, "step": 10}],
        }

    def test_log_artifact(self):
        exp = Experiment(id="exp4", name="Artifact Test")
        exp.log_artifact("/path/to/model.pt", "trained_model")
        exp.log_artifact("/path/to/results.csv")
        assert exp.artifacts == [
            {"path": "/path/to/model.pt", "name": "trained_model"},
            {"path": "/path/to/results.csv", "name": "results.csv"},
        ]

    def test_save_and_load_experiment(self):
        exp = Experiment(
            id="exp5", name="Save/Load Test", description="Testing persistence"
        )
        exp.log_hyperparameter("epochs", 5)
        exp.log_metric("val_loss", 0.05, step=100)
        exp.log_artifact("/tmp/chart.png")

        exp.save(self.test_dir)

        loaded_exp = Experiment.load("exp5", self.test_dir)

        assert loaded_exp.id == exp.id
        assert loaded_exp.name == exp.name
        assert loaded_exp.description == exp.description
        assert loaded_exp.hyperparameters == exp.hyperparameters
        assert loaded_exp.metrics == exp.metrics
        assert loaded_exp.artifacts == exp.artifacts

    def test_load_nonexistent_experiment(self):
        with pytest.raises(FileNotFoundError, match="Experiment file not found"):
            Experiment.load("nonexistent_exp", self.test_dir)
