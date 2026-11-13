"""Train/eval deep RSD heads on cached CNN features."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data_provider.datasets.feature_rsd import FeatureRSD
from exp.exp_basic import ExpBasic
from utils.io import ensure_dir, save_json, save_predictions_csv, save_summary
from utils.metrics import compute_dataset_metrics
from utils.seed import set_seed
from utils.splits import build_protocol_splits, save_splits


class ExpFeatureRSD(ExpBasic):
    def __init__(self, configs: Mapping[str, Any], repo_root):
        super().__init__(configs)
        self.repo_root = repo_root
        self.splits = build_protocol_splits(configs, repo_root)
        ensure_dir(self.output_dir)
        save_splits(self.splits, self.output_dir / "splits.json")
        self.feature_dir = configs.get("feature_dir")
        if not self.feature_dir:
            raise ValueError("feature_dir required for ExpFeatureRSD")
        mcfg = configs.get("model") or {}
        self.max_len = float(mcfg.get("max_len", mcfg.get("rsd_norm", 90.0)))

    def _loader_ids(self, ids, shuffle=False):
        ds = FeatureRSD(self.feature_dir, ids)
        return torch.utils.data.DataLoader(
            ds,
            batch_size=1,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=lambda xs: xs[0],
        )

    def _loss(self, out, batch):
        pred = out["rsd"].reshape(-1)
        tgt = batch["rsd"].to(pred.device).reshape(-1)
        kind = str(self.configs.get("feat_loss", "rsdbench"))
        if kind == "paper_timelstm":
            scale = max(self.max_len, 1e-6)
            p = (pred / scale).clamp(1e-6, 1 - 1e-6)
            return F.binary_cross_entropy(p, (tgt / scale).clamp(0, 1))
        if kind == "paper_rsdnet":
            scale = max(self.max_len, 1e-6)
            loss = F.smooth_l1_loss(pred / scale, tgt / scale)
            return loss + F.smooth_l1_loss(out["progress"].reshape(-1), batch["progress"].to(pred.device).reshape(-1))
        if kind != "rsdbench":
            raise ValueError(f"feat_loss={kind!r}")
        # Normalize by dataset max_len for stable gradients across short/long cases
        pred_n = pred / max(self.max_len, 1e-6)
        tgt_n = tgt / max(self.max_len, 1e-6)
        # Emphasize later part of surgery (clinically more actionable windows)
        # weight ~ 1 + 2*(1 - rsd/duration_proxy) using progress if available
        if "progress" in batch:
            prog = batch["progress"].to(pred.device).reshape(-1)
            w = (1.0 + 2.0 * prog).clamp(1.0, 3.0)
        else:
            w = torch.ones_like(pred_n)
        loss = (w * F.smooth_l1_loss(pred_n, tgt_n, reduction="none")).mean()

        if "progress" in out:
            loss = loss + F.smooth_l1_loss(
                out["progress"].reshape(-1), batch["progress"].to(pred.device).reshape(-1)
            )
        if "phase_logits" in out:
            logits = out["phase_logits"]
            if logits.dim() == 3:
                logits = logits.reshape(-1, logits.size(-1))
            else:
                logits = logits.reshape(-1, logits.size(-1))
            ph = batch["phase"].to(pred.device).reshape(-1)
            n = min(logits.size(0), ph.size(0))
            loss = loss + 0.5 * F.cross_entropy(logits[:n], ph[:n])
        if "future_phase_logits" in out and "phase" in batch:
            from layers.swag import build_future_phase_targets

            fut = out["future_phase_logits"]
            if fut.dim() == 4:
                fut = fut.reshape(-1, fut.size(-2), fut.size(-1))
            elif fut.dim() != 3:
                raise ValueError(f"unexpected future_phase_logits shape {tuple(fut.shape)}")
            ph = batch["phase"].to(fut.device).reshape(-1)
            eos_id = fut.size(-1) - 1
            horizon = fut.size(1)
            step = int((self.configs.get("model") or {}).get("future_step", 60))
            tgt_f = build_future_phase_targets(ph, horizon=horizon, eos_id=eos_id, step=step)
            loss = loss + 0.2 * F.cross_entropy(fut.reshape(-1, fut.size(-1)), tgt_f.reshape(-1))
        return loss

    def _train_one(self, model, train_ids, val_ids: Optional[list] = None):
        model.train()
        lr = float(self.configs.get("lr", 1e-3))
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=float(self.configs.get("weight_decay", 1e-4)),
        )
        epochs = int(self.configs.get("epochs", 40))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        loader = self._loader_ids(train_ids, shuffle=True)
        best_state = None
        best_val = float("inf")
        patience = int(self.configs.get("early_stopping", {}).get("patience", 12))
        bad = 0
        grad_clip = float(self.configs.get("grad_clip", 1.0))

        for ep in range(epochs):
            model.train()
            total, n = 0.0, 0
            for batch in loader:
                for k, v in list(batch.items()):
                    if torch.is_tensor(v):
                        batch[k] = v.to(self.device)
                opt.zero_grad(set_to_none=True)
                out = model(batch)
                loss = self._loss(out, batch)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                total += float(loss.item())
                n += 1
            sched.step()
            train_loss = total / max(n, 1)

            val_mae = None
            if val_ids:
                val_mae = float(self._eval(model, val_ids)["metrics"]["mae_all"])
                if val_mae < best_val - 1e-4:
                    best_val = val_mae
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
            if (ep + 1) % 5 == 0 or ep == 0:
                msg = f"[feat-train] epoch={ep+1}/{epochs} loss={train_loss:.4f}"
                if val_mae is not None:
                    msg += f" val_mae_all={val_mae:.3f} best={best_val:.3f}"
                print(msg)
            if val_ids and bad >= patience:
                print(f"[early-stop] epoch={ep+1} best_val_mae_all={best_val:.3f}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(self.device)
        return model

    @torch.no_grad()
    def _eval(self, model, video_ids):
        model.eval()
        loader = self._loader_ids(video_ids, shuffle=False)
        per_video, rows = {}, []
        for batch in loader:
            for k, v in list(batch.items()):
                if torch.is_tensor(v):
                    batch[k] = v.to(self.device)
            out = model(batch)
            pred = out["rsd"].detach().cpu().numpy().reshape(-1)
            w = int(self.configs.get("test_smooth_window", 0) or 0)
            if w > 1:
                # Causal moving average over the last w predictions (TimeLSTM: 15 s window).
                c = np.cumsum(np.concatenate([[0.0], pred]))
                idx = np.arange(1, len(pred) + 1)
                pred = (c[idx] - c[np.maximum(idx - w, 0)]) / np.minimum(idx, w)
            tgt = batch["rsd"].detach().cpu().numpy().reshape(-1)
            elapsed = batch["elapsed"].detach().cpu().numpy().reshape(-1)
            vid = str(batch["video_id"])
            per_video[vid] = {"pred": pred, "target": tgt}
            for j in range(len(pred)):
                rows.append(
                    {
                        "video_id": vid,
                        "t_idx": j,
                        "elapsed_min": float(elapsed[j]),
                        "rsd_true": float(tgt[j]),
                        "rsd_pred": float(pred[j]),
                    }
                )
        from utils.metrics import protocol_windows

        metrics = compute_dataset_metrics(per_video, windows=protocol_windows(self.configs))
        return {"metrics": metrics, "rows": rows}

    def run(self) -> Dict[str, Any]:
        scheme = self.splits.get("scheme")
        repeat_metrics = {}
        if scheme in ("repeated_holdout", "stratified_repeated_holdout"):
            for rep in self.splits["repeats"]:
                rid = f"seed{rep['seed']}"
                done = self.output_dir / rid / "metrics.json"
                if done.exists():
                    print(f"[skip] {rid} already has metrics.json", flush=True)
                    repeat_metrics[rid] = json.loads(done.read_text())
                    continue
                set_seed(int(rep["seed"]))
                model = self._build_model()
                train_ids = list(rep["train"])
                val_ids = list(rep.get("val") or [])
                model = self._train_one(model, train_ids, val_ids=val_ids or None)
                result = self._eval(model, rep["test"])
                ensure_dir(self.output_dir / rid)
                torch.save(model.state_dict(), self.output_dir / rid / "model.pth")
                save_predictions_csv(result["rows"], self.output_dir / rid / "predictions.csv")
                save_json(result["metrics"], self.output_dir / rid / "metrics.json")
                repeat_metrics[rid] = result["metrics"]
                print(f"[{rid}] mae_all={result['metrics']['mae_all']:.3f} mae_5={result['metrics']['mae_5']:.3f}")
        elif scheme == "fixed_holdout_inner_kfold":
            for fold in self.splits["folds"]:
                fid = fold["fold"]
                set_seed(int(self.configs.get("split", {}).get("seed_start", 42)) + fold["fold_index"])
                model = self._build_model()
                # carve a small val from train for early stopping
                train_ids = list(fold["train"])
                n_val = max(1, len(train_ids) // 5)
                val_ids = train_ids[-n_val:]
                train_ids = train_ids[:-n_val]
                model = self._train_one(model, train_ids, val_ids=val_ids)
                result = self._eval(model, fold["test"])
                ensure_dir(self.output_dir / fid)
                torch.save(model.state_dict(), self.output_dir / fid / "model.pth")
                save_predictions_csv(result["rows"], self.output_dir / fid / "predictions.csv")
                save_json(result["metrics"], self.output_dir / fid / "metrics.json")
                repeat_metrics[fid] = result["metrics"]
                print(f"[{fid}] mae_all={result['metrics']['mae_all']:.3f} mae_5={result['metrics']['mae_5']:.3f}")
        else:
            raise ValueError(scheme)
        return save_summary(self.output_dir, repeat_metrics)
