"""Experiment loop for statistical baselines (no gradients)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np
import torch

from data_provider.data_factory import build_dataset
from exp.exp_basic import ExpBasic
from utils.io import ensure_dir, save_json, save_predictions_csv, save_summary
from utils.metrics import compute_dataset_metrics
from utils.splits import build_protocol_splits, save_splits


class ExpBaseline(ExpBasic):
    def __init__(self, configs: Mapping[str, Any], repo_root):
        super().__init__(configs)
        self.repo_root = repo_root
        self.splits = build_protocol_splits(configs, repo_root)
        ensure_dir(self.output_dir)
        save_splits(self.splits, self.output_dir / "splits.json")
        self.model = self._build_model()

    def _fit_on_train(self, train_ids):
        ds = build_dataset(self.configs, train_ids, split="train", return_frames=False)
        metas = list(ds.iter_meta())
        durations = [m["duration_min"] for m in metas]
        if hasattr(self.model, "fit"):
            self.model.fit(np.asarray(durations, dtype=np.float64))
        if hasattr(self.model, "fit_from_videos"):
            phases = [m["phase"] for m in metas]
            elapsed = [m["elapsed"] for m in metas]
            tools = [m.get("tools") for m in metas]
            groups = [m.get("surgeon") for m in metas]
            experiences = [m.get("surgeon_exp") for m in metas]
            try:
                self.model.fit_from_videos(
                    phases,
                    elapsed,
                    durations,
                    groups=groups,
                    experiences=experiences,
                    tools_list=tools,
                )
            except TypeError:
                try:
                    self.model.fit_from_videos(
                        phases, elapsed, durations, groups=groups, experiences=experiences
                    )
                except TypeError:
                    self.model.fit_from_videos(phases, elapsed, durations)

    @torch.no_grad()
    def _eval_split(self, video_ids) -> Dict[str, Any]:
        ds = build_dataset(self.configs, video_ids, split="test", return_frames=False)
        per_video = {}
        rows = []
        for meta in ds.iter_meta():
            batch = {
                "elapsed": torch.from_numpy(meta["elapsed"]),
                "phase": torch.from_numpy(meta["phase"]),
                "rsd": torch.from_numpy(meta["rsd"]),
            }
            if meta.get("tools") is not None:
                batch["tools"] = torch.from_numpy(np.asarray(meta["tools"], dtype=np.float32))
            if meta.get("surgeon") is not None:
                batch["surgeon"] = meta["surgeon"]
            if meta.get("surgeon_exp") is not None:
                batch["surgeon_exp"] = meta["surgeon_exp"]
            out = self.model(batch)
            pred = out["rsd"].detach().cpu().numpy().reshape(-1)
            tgt = meta["rsd"].reshape(-1)
            vid = str(meta["video_id"])
            per_video[vid] = {"pred": pred, "target": tgt}
            for i, (p, t, e) in enumerate(zip(pred, tgt, meta["elapsed"])):
                rows.append(
                    {
                        "video_id": vid,
                        "t_idx": i,
                        "elapsed_min": float(e),
                        "rsd_true": float(t),
                        "rsd_pred": float(p),
                    }
                )
        from utils.metrics import protocol_windows

        metrics = compute_dataset_metrics(per_video, windows=protocol_windows(self.configs))
        return {"metrics": metrics, "rows": rows, "per_video": per_video}

    def run(self) -> Dict[str, Any]:
        scheme = self.splits.get("scheme")
        repeat_metrics: Dict[str, Dict[str, float]] = {}

        if scheme in ("repeated_holdout", "stratified_repeated_holdout"):
            from utils.seed import set_seed

            for rep in self.splits["repeats"]:
                rid = f"seed{rep['seed']}"
                set_seed(int(rep["seed"]))
                self.model = self._build_model()
                # Fit on train+val pool (baselines have no early-stop); test never seen.
                self._fit_on_train(rep["train"] + (rep.get("val") or []))
                result = self._eval_split(rep["test"])
                save_predictions_csv(result["rows"], self.output_dir / rid / "predictions.csv")
                save_json(result["metrics"], self.output_dir / rid / "metrics.json")
                repeat_metrics[rid] = result["metrics"]
        elif scheme == "fixed_holdout_inner_kfold":
            # LEGACY appendix only (CataNet fixed test + folds)
            fold_metrics = {}
            for fold in self.splits["folds"]:
                fid = fold["fold"]
                self.model = self._build_model()
                self._fit_on_train(fold["train"])
                result = self._eval_split(fold["test"])
                save_predictions_csv(result["rows"], self.output_dir / fid / "predictions.csv")
                save_json(result["metrics"], self.output_dir / fid / "metrics.json")
                fold_metrics[fid] = result["metrics"]
            repeat_metrics = fold_metrics
        else:
            raise ValueError(scheme)

        return save_summary(self.output_dir, repeat_metrics)
