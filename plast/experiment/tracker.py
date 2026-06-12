import os
import time
import logging
from datetime import datetime
from .yaml_utils import dump_yaml, load_yaml

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Tracks metrics and checkpoints for a training run.

    Each :class:`ExperimentTracker` corresponds to one numbered run under
    ``experiments/<name>/run_NNN/``.  The directory structure is created
    automatically::

        experiments/
        └── mlp_mnist/
            ├── summary.yaml            ← best run across all runs
            ├── run_001/
            │   ├── config.yaml         ← frozen config
            │   ├── metrics.yaml        ← per-epoch metrics
            │   └── checkpoints/
            │       └── best_model.npz  ← best weights

    Example::

        config  = ExperimentConfig("mlp_mnist", model={}, training={"lr": 1e-3})
        tracker = ExperimentTracker(config)

        for epoch in range(50):
            loss, acc = train_epoch(model, loader)
            tracker.log_epoch(epoch, {"train_loss": loss, "val_accuracy": acc}, model=model)

        tracker.finish()

    Args:
        config:   An :class:`~plast.experiment.ExperimentConfig` instance.
        base_dir: Root directory for all experiments (default ``./experiments``).
        verbose:  If ``True``, print a summary line on each ``log_epoch`` call
                  (default ``True``).
    """

    def __init__(self, config, base_dir: str = "./experiments", verbose: bool = True):
        self.config = config
        self.base_dir = base_dir
        self.verbose = verbose
        self.experiment_dir = os.path.join(base_dir, config.name)

        # Determine run ID (run_001, run_002, …)
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
                except (ValueError, IndexError):
                    pass
            next_run_num = max(run_nums) + 1 if run_nums else 1
        else:
            next_run_num = 1

        self.run_id = f"run_{next_run_num:03d}"
        self.run_dir = os.path.join(self.experiment_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Freeze config
        self.config.save(os.path.join(self.run_dir, "config.yaml"))

        # State
        self.started_at = datetime.now().isoformat()
        self.start_time = time.time()
        self.epochs_log = []
        self.best_accuracy = -1.0
        self.best_loss = float("inf")
        self.best_epoch = -1
        self.best_metrics = {}

        logger.info("Experiment '%s' started — run %s", config.name, self.run_id)

    def log_epoch(self, epoch: int, metrics: dict, model=None) -> bool:
        """Record metrics for one epoch and optionally checkpoint the model.

        Args:
            epoch:   Current epoch number (0-indexed).
            metrics: Dict of metric names to scalar values.  The keys
                     ``'val_accuracy'`` and ``'val_loss'`` are used to
                     determine the best run automatically.
            model:   If provided **and** this is the best epoch so far, the
                     model's weights are saved to a ``.npz`` checkpoint.

        Returns:
            ``True`` if this was a new best epoch, ``False`` otherwise.
        """
        metrics = dict(metrics)
        metrics["epoch"] = epoch
        metrics["timestamp"] = datetime.now().isoformat()
        self.epochs_log.append(metrics)

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

        if is_best and model is not None:
            import numpy as np

            state = model.state_dict()
            checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.npz")
            np.savez(checkpoint_path, **state)
            logger.info("Epoch %d: new best — checkpoint saved to %s", epoch, checkpoint_path)

        if self.verbose:
            parts = [f"epoch={epoch}"]
            for k, v in metrics.items():
                if k in ("epoch", "timestamp"):
                    continue
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v}")
            star = " ★" if is_best else ""
            print(f"[{self.config.name}/{self.run_id}] {' | '.join(parts)}{star}")

        self._write_metrics()
        return is_best

    def _write_metrics(self) -> None:
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

    def finish(self) -> None:
        """Finalize the run: write metrics and update the experiment summary."""
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

        self._update_summary(total_duration)

        mins, secs = divmod(total_duration, 60)
        duration_str = f"{mins}m {secs}s" if mins else f"{secs}s"
        if self.verbose:
            best_str = ""
            if self.best_epoch >= 0:
                if self.best_accuracy >= 0:
                    best_str = f" | best val_acc={self.best_accuracy:.4f} @ epoch {self.best_epoch}"
                elif self.best_loss < float("inf"):
                    best_str = f" | best val_loss={self.best_loss:.4f} @ epoch {self.best_epoch}"
            print(
                f"[{self.config.name}/{self.run_id}] Finished in {duration_str}{best_str}"
            )

        logger.info(
            "Experiment '%s' run %s finished. Duration: %ds.",
            self.config.name,
            self.run_id,
            total_duration,
        )

    def _update_summary(self, total_duration: int) -> None:
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

    def __repr__(self) -> str:
        elapsed = int(time.time() - self.start_time)
        return (
            f"ExperimentTracker(name='{self.config.name}', run='{self.run_id}', "
            f"epochs={len(self.epochs_log)}, elapsed={elapsed}s)"
        )
