"""Rivoir 2020 training (Rivoir et al., MICCAI 2020)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from exp.exp_basic import ExpBasic
from exp.exp_rivoir_common import gpu_window, load_videos, read_frames
from utils.io import ensure_dir, save_json, save_predictions_csv, save_summary
from utils.metrics import compute_dataset_metrics, protocol_windows
from utils.seed import set_seed
from utils.splits import build_protocol_splits, save_splits


class ExpRivoir2020(ExpBasic):
    def __init__(self, configs, repo_root):
        super().__init__(configs)
        self.repo_root = repo_root
        self.splits = build_protocol_splits(configs, repo_root)
        ensure_dir(self.output_dir)
        save_splits(self.splits, self.output_dir / "splits.json")
        self.model = self._build_model()

    def _cfg(self):
        m = self.configs.get("model") or {}
        return {
            "lr": float(self.configs.get("lr", 1e-4)),
            "wd": float(self.configs.get("weight_decay", 1e-5)),
            "epochs": int(self.configs.get("epochs", 100)),
            "train_window": int(m.get("train_window", 256)),
            "lambda_ce": float(m.get("lambda_ce", 1e-2)),
            "patience": int((self.configs.get("early_stopping") or {}).get("patience", 15)),
            "grad_accum": int(m.get("grad_accum", 1)),
            "batch_windows": int(m.get("batch_windows", 6)),  # pack several clips / step → ~VRAM
        }

    def _window(self, T, win):
        if T <= win:
            return 0, T
        s = int(np.random.randint(0, T - win + 1))
        return s, s + win

    def _train_one(self, model, train_ids, val_ids=None):
        cfg = self._cfg()
        ids_all = list(dict.fromkeys(list(train_ids) + list(val_ids or [])))
        videos = load_videos(self.configs, ids_all)
        self._videos = videos
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        ckpt = Path(getattr(self, "_ckpt_dir"))
        ensure_dir(ckpt)
        best_p, last_p = ckpt / "best.pt", ckpt / "last.pt"
        start, best, bad = 0, float("inf"), 0
        if last_p.exists() and bool(self.configs.get("resume", True)):
            st = torch.load(last_p, map_location=self.device)
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["optim"])
            start, best, bad = int(st["epoch"]), float(st["best"]), int(st["bad"])
            print(f"[resume] epoch={start} best={best:.4f}", flush=True)

        train_ids = list(train_ids)
        bw = max(1, cfg["batch_windows"])
        steps_per_ep = max(1, int(np.ceil(len(train_ids) / bw)) * 2)
        for ep in range(start, cfg["epochs"]):
            model.train()
            total, n = 0.0, 0
            for step in tqdm(range(steps_per_ep), desc=f"ep{ep+1}", leave=False):
                opt.zero_grad(set_to_none=True)
                loss_acc = 0.0
                for _ in range(bw):
                    vid = train_ids[int(np.random.randint(0, len(train_ids)))]
                    v = videos[vid]
                    frames = read_frames(v["frames"])  # CPU
                    elapsed = torch.from_numpy(v["elapsed"]).to(self.device)
                    rsd = torch.from_numpy(v["rsd"]).to(self.device)
                    phase = torch.from_numpy(v["phase"]).to(self.device)
                    s, e = self._window(frames.size(0), cfg["train_window"])
                    out = model({"frames": gpu_window(frames, s, e, self.device), "elapsed": elapsed[s:e]})
                    loss = F.smooth_l1_loss(out["rsd"], rsd[s:e])
                    loss = loss + cfg["lambda_ce"] * F.cross_entropy(out["phase_logits"], phase[s:e])
                    (loss / bw).backward()
                    loss_acc += float(loss.item())
                    del out
                opt.step()
                total += loss_acc / bw
                n += 1
            print(f"[train] epoch={ep+1} loss={total/max(n,1):.4f}", flush=True)
            if val_ids:
                score = float(self._eval(model, videos, val_ids)["metrics"]["mae_all"])
                improved = score < best
                if improved:
                    best, bad = score, 0
                    torch.save(model.state_dict(), best_p)
                else:
                    bad += 1
                print(f"[val] epoch={ep+1} mae_all={score:.4f} best={best:.4f} bad={bad}/{cfg['patience']}"
                      + (" **" if improved else ""), flush=True)
                torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "epoch": ep + 1,
                            "best": best, "bad": bad}, last_p)
                if cfg["patience"] > 0 and bad >= cfg["patience"]:
                    print(f"[early_stop] epoch={ep+1}", flush=True)
                    break
        if best_p.exists():
            model.load_state_dict(torch.load(best_p, map_location=self.device))
        return model

    @torch.no_grad()
    def _eval(self, model, videos, ids):
        model.eval()
        per, rows = {}, []
        for vid in ids:
            v = videos[vid]
            frames = read_frames(v["frames"])  # CPU
            elapsed = torch.from_numpy(v["elapsed"]).to(self.device)
            frames_g = frames.to(self.device, non_blocking=True)
            out = model({"frames": frames_g, "elapsed": elapsed})
            pred = out["rsd"].detach().cpu().numpy().reshape(-1)
            del frames_g, out
            tgt = v["rsd"].reshape(-1)
            per[str(vid)] = {"pred": pred, "target": tgt}
            for i in range(len(pred)):
                rows.append({"video_id": str(vid), "t_idx": i, "elapsed_min": float(v["elapsed"][i]),
                             "rsd_true": float(tgt[i]), "rsd_pred": float(pred[i])})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return {"metrics": compute_dataset_metrics(per, windows=protocol_windows(self.configs)), "rows": rows}

    def run(self) -> Dict[str, Any]:
        repeat_metrics = {}
        for rep in self.splits["repeats"]:
            rid = f"seed{rep['seed']}"
            done = self.output_dir / rid / "metrics.json"
            if done.exists():
                print(f"[skip] {rid}", flush=True)
                repeat_metrics[rid] = json.loads(done.read_text())
                continue
            set_seed(int(rep["seed"]))
            self.configs["seed"] = int(rep["seed"])
            model = self._build_model()
            self._ckpt_dir = self.output_dir / rid / "ckpt"
            ids_all = list(dict.fromkeys(list(rep["train"]) + list(rep.get("val") or []) + list(rep["test"])))
            videos = load_videos(self.configs, ids_all)
            model = self._train_one(model, rep["train"], rep.get("val"))
            result = self._eval(model, videos, rep["test"])
            ensure_dir(self.output_dir / rid)
            torch.save(model.state_dict(), self.output_dir / rid / "model.pth")
            save_predictions_csv(result["rows"], self.output_dir / rid / "predictions.csv")
            save_json(result["metrics"], self.output_dir / rid / "metrics.json")
            repeat_metrics[rid] = result["metrics"]
        return save_summary(self.output_dir, repeat_metrics)
