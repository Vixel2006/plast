import os
import tempfile
import numpy as np
import pytest
import plast


class TestExperimentConfig:
    def test_config_creation(self):
        config = plast.experiment.ExperimentConfig(
            name="test_run",
            model={"hidden_size": 64},
            training={"lr": 0.01, "epochs": 10},
            device="CPU",
        )
        assert config.name == "test_run"
        assert config.model["hidden_size"] == 64
        assert config.training["lr"] == 0.01

    def test_config_yaml_roundtrip(self):
        config = plast.experiment.ExperimentConfig(
            name="yaml_test",
            model={"type": "mlp", "layers": [64, 32]},
            training={"lr": 0.001, "optimizer": "Adam"},
            device="CUDA",
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config.save(f.name)
            f.flush()
            config_path = f.name

        loaded_config = plast.experiment.ExperimentConfig.load(config_path)
        assert loaded_config.name == "yaml_test"
        assert loaded_config.model["layers"] == [64, 32]
        assert loaded_config.training["optimizer"] == "Adam"

        os.unlink(config_path)


class TestExperimentTracker:
    def test_tracker_creation(self):
        config = plast.experiment.ExperimentConfig(
            name="tracker_test",
            model={"hidden_size": 8},
            training={"lr": 0.05, "epochs": 5},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = plast.experiment.ExperimentTracker(config, base_dir=tmpdir)
            tracker.finish()
            run_dir = tracker.run_dir
            assert os.path.exists(run_dir)
            assert os.path.exists(os.path.join(run_dir, "config.yaml"))

    def test_tracker_log_epoch(self):
        config = plast.experiment.ExperimentConfig(
            name="log_test",
            model={"hidden_size": 4},
            training={"lr": 0.01, "epochs": 3},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = plast.experiment.ExperimentTracker(config, base_dir=tmpdir)
            tracker.log_epoch(0, {"train_loss": 1.0, "val_loss": 1.0})
            tracker.log_epoch(1, {"train_loss": 0.5, "val_loss": 0.5})
            tracker.log_epoch(2, {"train_loss": 0.1, "val_loss": 0.1})
            tracker.finish()

            metrics_path = os.path.join(tracker.run_dir, "metrics.yaml")
            assert os.path.exists(metrics_path)

            summary_path = os.path.join(tmpdir, "log_test", "summary.yaml")
            assert os.path.exists(summary_path)

    def test_tracker_with_model_checkpoint(self):
        config = plast.experiment.ExperimentConfig(
            name="ckpt_test",
            model={"hidden_size": 8},
            training={"lr": 0.01, "epochs": 2},
        )

        plast.init_arenas(device=plast.Device.CPU)
        model = plast.nn.Sequential(
            plast.nn.Linear(2, 8),
            plast.nn.ReLU(),
            plast.nn.Linear(8, 1),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = plast.experiment.ExperimentTracker(config, base_dir=tmpdir)
            tracker.log_epoch(0, {"train_loss": 0.5, "val_loss": 0.5}, model=model)
            tracker.log_epoch(1, {"train_loss": 0.1, "val_loss": 0.1}, model=model)
            tracker.finish()

            best_path = os.path.join(tracker.checkpoint_dir, "best_model.npz")
            assert os.path.exists(best_path)

    def test_tracker_multiple_runs(self):
        config = plast.experiment.ExperimentConfig(
            name="multi_run",
            model={"hidden_size": 4},
            training={"lr": 0.01, "epochs": 1},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            t1 = plast.experiment.ExperimentTracker(config, base_dir=tmpdir)
            t1.finish()
            t2 = plast.experiment.ExperimentTracker(config, base_dir=tmpdir)
            t2.finish()
            assert t1.run_dir != t2.run_dir
