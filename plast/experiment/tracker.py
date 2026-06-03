import os
import time
from datetime import datetime
from .yaml_utils import dump_yaml, load_yaml


class ExperimentTracker:
    def __init__(self, config, base_dir="./experiments"):
        self.config = config
        self.base_dir = base_dir
        self.experiment_dir = os.path.join(base_dir, config.name)

        # 1. Determine run ID (run_001, run_002, ...)
        os.makedirs(self.experiment_dir, exist_ok=True)
        existing_runs = [
            d
            for d in os.listdir(self.experiment_dir)
            if d.startswith("run_") and os.path.isdir(os.path.join(self.experiment_dir, d))
        ]
        if existing_runs:
            run_nums = []
            for r in existing_runs:
                try:
                    run_nums.append(int(r.split("_")[1]))
                except ValueError:
                    pass
            next_run_num = max(run_nums) + 1 if run_nums else 1
        else:
            next_run_num = 1

        self.run_id = f"run_{next_run_num:03d}"
        self.run_dir = os.path.join(self.experiment_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 2. Freeze config to run directory
        self.config.save(os.path.join(self.run_dir, "config.yaml"))

        # Initialize metrics logs
        self.started_at = datetime.now().isoformat()
        self.start_time = time.time()
        self.epochs_log = []
        self.best_accuracy = -1.0
        self.best_loss = float("inf")
        self.best_epoch = -1
        self.best_metrics = {}

    def log_epoch(self, epoch, metrics, model=None):
        metrics = dict(metrics)
        metrics["epoch"] = epoch
        metrics["timestamp"] = datetime.now().isoformat()
        self.epochs_log.append(metrics)

        # Track best validation accuracy/loss
        val_acc = metrics.get("val_accuracy")
        val_loss = metrics.get("val_loss")
        is_best = False

        if val_acc is not None:
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.best_loss = val_loss if val_loss is not None else self.best_loss
                self.best_epoch = epoch
                self.best_metrics = metrics
                is_best = True
        elif val_loss is not None:
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_epoch = epoch
                self.best_metrics = metrics
                is_best = True

        # If best, save checkpoint
        if is_best and model is not None:
            import numpy as np

            state = model.state_dict()
            checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.npz")
            np.savez(checkpoint_path, **state)
            print(f"--> Saved best model checkpoint at epoch {epoch}")

        # Write incrementally to metrics.yaml
        self._write_metrics()

    def _write_metrics(self):
        metrics_data = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": None,
            "total_duration_seconds": int(time.time() - self.start_time),
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
            "epochs": self.epochs_log,
        }
        metrics_path = os.path.join(self.run_dir, "metrics.yaml")
        with open(metrics_path, "w") as f:
            f.write(dump_yaml(metrics_data))

    def finish(self):
        finished_at = datetime.now().isoformat()
        total_duration = int(time.time() - self.start_time)

        metrics_data = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "total_duration_seconds": total_duration,
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
            "epochs": self.epochs_log,
        }
        metrics_path = os.path.join(self.run_dir, "metrics.yaml")
        with open(metrics_path, "w") as f:
            f.write(dump_yaml(metrics_data))

        # Update experiment-level summary.yaml
        self._update_summary(total_duration)
        print(
            f"Experiment {self.config.name} run {self.run_id} finished. Duration: {total_duration}s."
        )

    def _update_summary(self, total_duration):
        summary_path = os.path.join(self.experiment_dir, "summary.yaml")
        summary_data = {"experiment": self.config.name, "runs": []}
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r") as f:
                    summary_data = load_yaml(f.read())
            except Exception:
                pass

        run_summary = {
            "run_id": self.run_id,
            "best_val_accuracy": self.best_accuracy if self.best_accuracy >= 0 else None,
            "best_val_loss": self.best_loss if self.best_loss < float("inf") else None,
            "epochs_trained": len(self.epochs_log),
            "duration_seconds": total_duration,
        }

        runs = summary_data.get("runs", [])
        runs = [r for r in runs if r.get("run_id") != self.run_id]
        runs.append(run_summary)
        summary_data["runs"] = runs

        if self.best_accuracy >= 0:
            runs.sort(key=lambda x: x.get("best_val_accuracy") or 0.0, reverse=True)
        else:
            runs.sort(key=lambda x: x.get("best_val_loss") or float("inf"))

        if runs:
            summary_data["best_run"] = runs[0]

        with open(summary_path, "w") as f:
            f.write(dump_yaml(summary_data))
