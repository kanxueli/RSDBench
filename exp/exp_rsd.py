"""Training / evaluation loop for deep RSD models (skeleton)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data_provider.data_factory import build_dataloader, build_dataset
from exp.exp_basic import ExpBasic
from utils.io import ensure_dir, save_json, save_predictions_csv, save_summary
from utils.metrics import compute_dataset_metrics
from utils.seed import set_seed
from utils.splits import build_protocol_splits, save_splits


class ExpRSD(ExpBasic):
    def __init__(self, configs: Mapping[str, Any], repo_root):
        super().__init__(configs)
        self.repo_root = repo_root
        self.splits = build_protocol_splits(configs, repo_root)
        ensure_dir(self.output_dir)
        save_splits(self.splits, self.output_dir / "splits.json")

    def _optimizer(self, model):
        lr = float(self.configs.get("lr", 1e-4))
        wd = float(self.configs.get("weight_decay", 0.0))
        mcfg = self.configs.get("model") or {}
        cnn_mult = float(self.configs.get("cnn_lr_mult", mcfg.get("cnn_lr_mult", 1.0)))
        # Optional lower LR for ImageNet CNN backbone (TransLocal / similar).
        cnn = getattr(model, "cnn", None) or getattr(model, "backbone", None)
        if cnn is not None and abs(cnn_mult - 1.0) > 1e-12:
            cnn_ids = {id(p) for p in cnn.parameters()}
            cnn_params = [p for p in model.parameters() if p.requires_grad and id(p) in cnn_ids]
            other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in cnn_ids]
            groups = []
            if other_params:
                groups.append({"params": other_params, "lr": lr})
            if cnn_params:
                groups.append({"params": cnn_params, "lr": lr * cnn_mult})
            if groups:
                return torch.optim.Adam(groups, weight_decay=wd)
        return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)

    def _loss(self, out, batch):
        pred = out["rsd"].reshape(-1)
        tgt = batch["rsd"].to(pred.device).reshape(-1)
        loss = F.l1_loss(pred, tgt)
        if "phase_logits" in out and "phase" in batch:
            logits = out["phase_logits"]
            if logits.dim() == 3:
                logits = logits.reshape(-1, logits.size(-1))
            else:
                logits = logits.reshape(-1, logits.size(-1))
            ph = batch["phase"].to(pred.device).reshape(-1).long()
            n = min(logits.size(0), ph.size(0))
            loss = loss + 0.5 * F.cross_entropy(logits[:n], ph[:n])
        return loss

    def _maybe_window(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Random contiguous clip for memory (CataNet-style 48-frame windows)."""
        win = int(self.configs.get("train_window", 0) or (self.configs.get("model") or {}).get("train_window", 0) or 0)
        if win <= 0 or "frames" not in batch:
            return batch
        t = batch["frames"].shape[0] if batch["frames"].dim() == 4 else batch["frames"].shape[1]
        if t <= win:
            return batch
        start = int(torch.randint(0, t - win + 1, (1,)).item())
        end = start + win
        out = dict(batch)
        for k, v in batch.items():
            if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == t:
                out[k] = v[start:end]
            elif torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == t:
                out[k] = v[:, start:end]
        return out

    def _es_cfg(self) -> Dict[str, Any]:
        es = self.configs.get("early_stopping") or {}
        return {
            "enabled": bool(es.get("enabled", True)),
            "monitor": str(es.get("monitor", "mae_all")),
            "patience": int(es.get("patience", 10)),
            "min_epochs": int(es.get("min_epochs", 0)),
            "val_every": int(es.get("val_every", 1)),
        }

    def _train_one(self, model, train_ids, val_ids=None, ckpt_dir=None):
        """Train with per-epoch val (full-video MAE), best/last ckpt, early stop, resume.

        ckpt_dir/last.pt  : model + optimizer + epoch + ES state (resume after crash)
        ckpt_dir/best.pt  : best-val weights; reloaded before test
        """
        model.train()
        ds = build_dataset(self.configs, train_ids, split="train", return_frames=True)
        loader = build_dataloader(ds, self.configs, shuffle=True)
        opt = self._optimizer(model)
        epochs = int(self.configs.get("epochs", 10))
        grad_clip = float(self.configs.get("grad_clip", 0.0) or 0.0)
        es = self._es_cfg()
        use_val = bool(val_ids) and es["enabled"]
        ckpt_dir = ckpt_dir or getattr(self, "_ckpt_dir", None)
        ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        if ckpt_dir is not None:
            ensure_dir(ckpt_dir)
        last_p = ckpt_dir / "last.pt" if ckpt_dir else None
        best_p = ckpt_dir / "best.pt" if ckpt_dir else None

        start_ep, best, bad, history = 0, float("inf"), 0, []
        if last_p is not None and last_p.exists() and bool(self.configs.get("resume", True)):
            st = torch.load(last_p, map_location=self.device)
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["optim"])
            start_ep = int(st["epoch"])
            best = float(st.get("best", float("inf")))
            bad = int(st.get("bad", 0))
            history = list(st.get("history", []))
            print(f"[resume] {last_p} epoch={start_ep} best={best:.4f} bad={bad}", flush=True)
            if st.get("stopped", False):
                start_ep = epochs

        for ep in range(start_ep, epochs):
            model.train()
            total = 0.0
            n = 0
            for batch in tqdm(loader, desc=f"epoch {ep+1}/{epochs}", leave=False):
                batch = self._maybe_window(batch)
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
            rec = {"epoch": ep + 1, "train_loss": total / max(n, 1)}
            print(f"[train] epoch={ep+1} loss={rec['train_loss']:.4f}", flush=True)

            stop = False
            if use_val and ((ep + 1) % max(es["val_every"], 1) == 0 or ep + 1 == epochs):
                vm = self._eval(model, val_ids, split="val")["metrics"]
                score = float(vm[es["monitor"]])
                rec["val_" + es["monitor"]] = score
                improved = score < best
                if improved:
                    best, bad = score, 0
                    if best_p is not None:
                        torch.save(model.state_dict(), best_p)
                else:
                    bad += 1
                print(
                    f"[val] epoch={ep+1} {es['monitor']}={score:.4f} best={best:.4f} bad={bad}/{es['patience']}"
                    + ("  **" if improved else ""),
                    flush=True,
                )
                if es["patience"] > 0 and bad >= es["patience"] and ep + 1 >= es["min_epochs"]:
                    print(f"[early_stop] epoch={ep+1} best={best:.4f}", flush=True)
                    stop = True
            history.append(rec)
            if last_p is not None:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optim": opt.state_dict(),
                        "epoch": ep + 1,
                        "best": best,
                        "bad": bad,
                        "history": history,
                        "stopped": stop,
                    },
                    last_p,
                )
            if stop:
                break

        if best_p is not None and best_p.exists() and use_val:
            model.load_state_dict(torch.load(best_p, map_location=self.device))
            print(f"[best] loaded {best_p} ({es['monitor']}={best:.4f})", flush=True)
        if ckpt_dir is not None:
            save_json({"history": history, "best": best, "monitor": es["monitor"]}, ckpt_dir / "history.json")
        return model

    @torch.no_grad()
    def _eval(self, model, video_ids, split: str = "test"):
        model.eval()
        ds = build_dataset(self.configs, video_ids, split=split, return_frames=True)
        per_video = {}
        rows = []
        for i in range(len(ds)):
            batch = ds[i]
            for k, v in list(batch.items()):
                if torch.is_tensor(v):
                    batch[k] = v.to(self.device)
            out = model(batch)
            pred = out["rsd"].detach().cpu().numpy().reshape(-1)
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
                # Per-repeat seed so models can resolve FT / Siamese ckpts under seed{N}/
                self.configs["seed"] = int(rep["seed"])
                if isinstance(self.configs.get("model"), dict):
                    self.configs["model"]["seed"] = int(rep["seed"])
                model = self._build_model()
                # Attribute, not kwarg: subclasses (PitRSDNet, SWAG) override _train_one(model, train, val).
                self._ckpt_dir = self.output_dir / rid / "ckpt"
                model = self._train_one(model, rep["train"], rep.get("val"))
                self._ckpt_dir = None
                result = self._eval(model, rep["test"])
                ensure_dir(self.output_dir / rid)
                torch.save(model.state_dict(), self.output_dir / rid / "model.pth")
                save_predictions_csv(result["rows"], self.output_dir / rid / "predictions.csv")
                save_json(result["metrics"], self.output_dir / rid / "metrics.json")
                repeat_metrics[rid] = result["metrics"]
        elif scheme == "fixed_holdout_inner_kfold":
            fold_preds = []
            for fold in self.splits["folds"]:
                fid = fold["fold"]
                set_seed(int(self.configs.get("split", {}).get("seed_start", 42)) + fold["fold_index"])
                model = self._build_model()
                model = self._train_one(model, fold["train"], fold["val"])
                result = self._eval(model, fold["test"])
                ensure_dir(self.output_dir / fid)
                torch.save(model.state_dict(), self.output_dir / fid / "model.pth")
                save_predictions_csv(result["rows"], self.output_dir / fid / "predictions.csv")
                save_json(result["metrics"], self.output_dir / fid / "metrics.json")
                repeat_metrics[fid] = result["metrics"]
                fold_preds.append(result)
            # Optional: write ensemble-averaged predictions on fixed test
            if self.configs.get("split", {}).get("ensemble", True) and fold_preds:
                # average rsd_pred by (video_id, t_idx)
                from collections import defaultdict

                buckets = defaultdict(list)
                for res in fold_preds:
                    for row in res["rows"]:
                        buckets[(row["video_id"], row["t_idx"], row["elapsed_min"], row["rsd_true"])].append(
                            row["rsd_pred"]
                        )
                ens_rows = []
                for (vid, t_idx, e, t), preds in buckets.items():
                    ens_rows.append(
                        {
                            "video_id": vid,
                            "t_idx": t_idx,
                            "elapsed_min": e,
                            "rsd_true": t,
                            "rsd_pred": float(sum(preds) / len(preds)),
                        }
                    )
                save_predictions_csv(ens_rows, self.output_dir / "ensemble" / "predictions.csv")
        else:
            raise ValueError(scheme)

        return save_summary(self.output_dir, repeat_metrics)
