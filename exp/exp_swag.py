"""SWAG stage-2 training."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from exp.exp_rsd import ExpRSD
from utils.io import ensure_dir
from utils.metrics import compute_dataset_metrics, protocol_windows

def _parse_swag_stage2_log(log_path: Path) -> Optional[Dict[str, Any]]:
    if not log_path.is_file():
        return None
    text = log_path.read_text(errors="replace")
    parts = re.split(r"\[swag\] seed=", text)
    if len(parts) < 2:
        return None
    for last in reversed(parts[1:]):
        rows = []
        for m in re.finditer(
            r"\[swag\] epoch=(\d+)/\d+\s+lr=\S+\s+loss=(\S+)\s+val_mae_all=([\d.]+)",
            last,
        ):
            rows.append(
                {
                    "epoch": int(m.group(1)),
                    "train_loss": float(m.group(2)),
                    "val_mae_all": float(m.group(3)),
                }
            )
        if not rows:
            continue
        best = min(rows, key=lambda r: r["val_mae_all"])
        last_row = rows[-1]
        return {
            "best_ep": int(best["epoch"]),
            "best": float(best["val_mae_all"]),
            "last_ep": int(last_row["epoch"]),
            "bad": max(0, int(last_row["epoch"]) - int(best["epoch"])),
            "stopped": "[early_stop]" in last,
            "history": rows,
        }
    return None

DEFAULTS = {
    "ctx_length": 1440,
    "num_ctx_tokens": 12,
    "horizon_min": None,
    "loss_w_curr": 0.4,
    "loss_w_remaining_time": 1.0,
    "regression_loss_scale": 0.1,
    "rsd_loss_gamma": 10.0,
    "epochs": 80,
    "patience": 15,
    "warmup_epochs": 5,
    "batch_size": 64,
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "eta_min": 1e-7,
}

def remaining_time_capped(phase: np.ndarray, h: float, n_classes: int) -> np.ndarray:
    t_len = len(phase)
    out = np.full((t_len, n_classes + 1), float(h), dtype=np.float32)
    horizon = int(h * 60)
    for c in range(n_classes):
        nxt = -1
        for i in range(t_len - 1, -1, -1):
            if phase[i] == c:
                nxt = i
                out[i, c] = 0.0
            elif nxt >= 0 and nxt - i <= horizon:
                out[i, c] = (nxt - i) / 60.0
    eos_start = max(0, t_len - horizon)
    out[eos_start:, n_classes] = (t_len - np.arange(eos_start, t_len)) / 60.0
    return out

class ExpSWAG(ExpRSD):
    def _cfg(self) -> Dict[str, Any]:
        s = {**DEFAULTS, **dict(self.configs.get("swag_train") or {})}
        if s["horizon_min"] is None:
            s["horizon_min"] = {"cataract101": 20, "cholec80": 75}[self.configs["dataset"]]
        return s

    def _feature_dir(self, seed: int) -> Path:
        tmpl = self.configs.get("feature_dir") or str(
            Path(self.repo_root) / ".cache" / "features" / self.configs["dataset"] / "SWAG_vitLoViT_seed{seed}"
        )
        return Path(str(tmpl).format(seed=seed))

    def _load(self, ids, seed: int) -> Dict[Any, Dict[str, Any]]:
        d = self._feature_dir(seed)
        out = {}
        for vid in ids:
            obj = torch.load(d / f"{vid}.pt", map_location="cpu", weights_only=False)
            out[vid] = obj
        return out

    def _pack(self, videos: Dict[Any, Dict[str, Any]], ids, st, n_cls) -> Dict[str, torch.Tensor]:
        """Concatenate front-padded features of all videos on the device for fast window gathering."""
        L, dev = st["ctx_length"], self.device
        feats, phases, rts, starts, frame_t, vid_of = [], [], [], [], [], []
        off = 0
        for k, vid in enumerate(ids):
            v = videos[vid]
            f = v["features"].float()
            T = f.size(0)
            feats.append(torch.cat([torch.zeros(L, f.size(1)), f]))
            ph = v["phase"].long()
            phases.append(torch.cat([torch.full((st["num_ctx_tokens"] - 1,), -1, dtype=torch.long), ph]))
            rts.append(torch.from_numpy(remaining_time_capped(ph.numpy(), st["horizon_min"], n_cls)))
            starts.append(torch.full((T,), off, dtype=torch.long))
            frame_t.append(torch.arange(T))
            vid_of.append(torch.full((T,), k, dtype=torch.long))
            off += T + L
        ph_off = torch.cumsum(torch.tensor([0] + [len(p) for p in phases[:-1]]), 0)
        return {
            "feats": torch.cat(feats).to(dev),
            "phases": torch.cat(phases).to(dev),
            "ph_off": ph_off.to(dev),
            "rt": torch.cat(rts).to(dev),
            "start": torch.cat(starts).to(dev),
            "t": torch.cat(frame_t).to(dev),
            "vid": torch.cat(vid_of).to(dev),
        }

    def _batch(self, pk, idx: torch.Tensor, st):
        L, K = st["ctx_length"], st["num_ctx_tokens"]
        t, start = pk["t"][idx], pk["start"][idx]
        win = start[:, None] + t[:, None] + torch.arange(L, device=idx.device)[None]  # features [t-L, t)
        x = pk["feats"][win]
        cur = pk["ph_off"][pk["vid"][idx]][:, None] + t[:, None] + torch.arange(K, device=idx.device)[None]
        return x, pk["phases"][cur], pk["rt"][idx].unsqueeze(1)

    def _class_weights(self, videos, train_ids, n_cls) -> torch.Tensor:
        counts = np.zeros(n_cls)
        for vid in train_ids:
            counts += np.bincount(videos[vid]["phase"].numpy(), minlength=n_cls)[:n_cls]
        present = counts > 0
        w = np.zeros(n_cls)
        w[present] = counts.sum() / (present.sum() * counts[present])
        return torch.tensor(w, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _predict_rsd(self, model, pk, st, n_cls, bs=256) -> torch.Tensor:
        model.eval()
        preds = []
        for s in range(0, pk["t"].numel(), bs):
            idx = torch.arange(s, min(s + bs, pk["t"].numel()), device=self.device)
            x, _, _ = self._batch(pk, idx, st)
            preds.append(model(x)["remaining_time"][:, 0, n_cls].float())
        return torch.cat(preds)

    def _per_video(self, pred: torch.Tensor, pk, videos, ids) -> Dict[str, Dict[str, np.ndarray]]:
        pred = pred.cpu().numpy()
        vid_idx = pk["vid"].cpu().numpy()
        return {
            str(vid): {"pred": pred[vid_idx == k], "target": videos[vid]["rsd"].numpy()}
            for k, vid in enumerate(ids)
        }

    def _train_one(self, model, train_ids, val_ids=None):
        st = self._cfg()
        seed = int(self.configs.get("seed", 42))
        n_cls = model.n_phases
        out_dir = ensure_dir(self.output_dir / f"seed{seed}")
        val_ids = list(val_ids or [])
        videos = self._load(list(train_ids) + val_ids, seed)
        pk_tr = self._pack(videos, list(train_ids), st, n_cls)
        pk_va = self._pack(videos, val_ids, st, n_cls) if val_ids else None
        cw = self._class_weights(videos, train_ids, n_cls)
        n = pk_tr["t"].numel()
        steps_per_epoch = math.ceil(n / st["batch_size"])
        warm, total = st["warmup_epochs"] * steps_per_epoch, st["epochs"] * steps_per_epoch
        opt = torch.optim.SGD(model.parameters(), lr=st["lr"], momentum=0.9, nesterov=True,
                              weight_decay=st["weight_decay"])

        def lr_at(step):
            if step < warm:
                return st["lr"] * step / warm
            p = (step - warm) / max(1, total - warm)
            return st["eta_min"] + 0.5 * (st["lr"] - st["eta_min"]) * (1 + math.cos(math.pi * p))

        best_p, last_p = out_dir / "swag_best.pth", out_dir / "last.pt"
        best, step, bad, history = float("inf"), 0, 0, []
        patience = int(st.get("patience", 0))
        gen = torch.Generator(device="cpu").manual_seed(seed)
        start_ep = 0
        do_resume = bool(self.configs.get("resume", True))
        resume_msg = None
        if do_resume and last_p.exists():
            ck = torch.load(last_p, map_location=self.device, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optim"])
            start_ep = int(ck["epoch"])
            step = int(ck.get("step", start_ep * steps_per_epoch))
            best = float(ck.get("best", float("inf")))
            bad = int(ck.get("bad", 0))
            history = list(ck.get("history", []))
            if "gen" in ck:
                gs = ck["gen"].detach().to("cpu").contiguous()
                if not isinstance(gs, torch.ByteTensor):
                    gs = torch.ByteTensor(gs.cpu().numpy())
                gen.set_state(gs)
            else:
                for _ in range(start_ep):
                    torch.randperm(n, generator=gen)
            resume_msg = f"[resume] {last_p} epoch={start_ep} step={step} best={best:.4f} bad={bad}"
            if ck.get("stopped", False):
                start_ep = int(st["epochs"])
        elif do_resume and best_p.exists():
            log_candidates = [
                Path(self.repo_root) / "results" / "logs" / f"cholec80_seed{seed}_swag_stage2.log",
                Path(self.repo_root) / "results" / "logs" / f"{self.configs['dataset']}_seed{seed}_swag_stage2.log",
                out_dir / "train.log",
            ]
            parsed = None
            log_used = None
            for lp in log_candidates:
                parsed = _parse_swag_stage2_log(lp)
                if parsed is not None:
                    log_used = lp
                    break
            if parsed is not None:
                model.load_state_dict(torch.load(best_p, map_location=self.device, weights_only=False))
                start_ep = int(parsed["best_ep"])
                best = float(parsed["best"])
                bad = int(parsed["bad"])
                step = start_ep * steps_per_epoch
                history = list(parsed.get("history") or [])
                for _ in range(start_ep):
                    torch.randperm(n, generator=gen)
                resume_msg = (
                    f"[resume] bootstrap {best_p.name} + {log_used.name}: "
                    f"best@ep{parsed['best_ep']}={best:.4f} last_ep={parsed['last_ep']} "
                    f"→ start_ep={start_ep} bad={bad}/{patience} (fresh optim)"
                )
                if parsed.get("stopped"):
                    start_ep = int(st["epochs"])
            else:
                resume_msg = f"[resume] found {best_p} but no parseable stage-2 log → train from scratch"
        print(f"[swag] seed={seed} H={st['horizon_min']} train_frames={n} steps/epoch={steps_per_epoch} cfg={st}",
              flush=True)
        if resume_msg:
            print(resume_msg, flush=True)
        H = float(st["horizon_min"])
        for ep in range(start_ep, st["epochs"]):
            model.train()
            perm = torch.randperm(n, generator=gen).to(self.device)
            tot = []
            for s in range(0, n, st["batch_size"]):
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                x, cur_t, rt_t = self._batch(pk_tr, perm[s : s + st["batch_size"]], st)
                out = model(x)
                ce = F.cross_entropy(out["curr_frames"].permute(0, 2, 1), cur_t, weight=cw,
                                     ignore_index=-1, reduction="none").mean()
                w = 1 + torch.exp(-st["rsd_loss_gamma"] * rt_t / H)
                rt_loss = ((out["remaining_time"] - rt_t) ** 2 * w).mean()
                loss = st["loss_w_curr"] * ce + st["loss_w_remaining_time"] * st["regression_loss_scale"] * rt_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                step += 1
                tot.append(float(loss.item()))
            mark = ""
            stop = False
            if pk_va is not None:
                pv = self._per_video(self._predict_rsd(model, pk_va, st, n_cls), pk_va, videos, val_ids)
                v = compute_dataset_metrics(pv, windows=protocol_windows(self.configs))["mae_all"]
                if v < best:
                    best, bad = v, 0
                    torch.save(model.state_dict(), best_p)
                    mark = "  **"
                else:
                    bad += 1
                    mark = f"  bad={bad}/{patience}" if patience > 0 else ""
                if patience > 0 and bad >= patience:
                    stop = True
            else:
                v = float("nan")
                torch.save(model.state_dict(), best_p)
            train_loss = float(np.mean(tot))
            history.append({"epoch": ep + 1, "train_loss": train_loss, "val_mae_all": v, "lr": lr_at(step),
                            "best": best, "bad": bad})
            print(f"[swag] epoch={ep+1}/{st['epochs']} lr={lr_at(step):.3g} loss={train_loss:.4f} "
                  f"val_mae_all={v:.4f}{mark}", flush=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": opt.state_dict(),
                    "epoch": ep + 1,
                    "step": step,
                    "best": best,
                    "bad": bad,
                    "history": history,
                    "stopped": stop,
                    "gen": gen.get_state(),
                    "seed": seed,
                },
                last_p,
            )
            if stop:
                print(f"[early_stop] swag epoch={ep+1} best_val={best:.4f}", flush=True)
                break
        if best_p.exists():
            model.load_state_dict(torch.load(best_p, map_location=self.device, weights_only=False))
            print(f"[best] loaded {best_p} (val_mae_all={best:.4f})", flush=True)
        return model

    @torch.no_grad()
    def _eval(self, model, video_ids):
        st = self._cfg()
        seed = int(self.configs.get("seed", 42))
        n_cls = model.n_phases
        ids = list(video_ids)
        videos = self._load(ids, seed)
        pk = self._pack(videos, ids, st, n_cls)
        per_video = self._per_video(self._predict_rsd(model, pk, st, n_cls), pk, videos, ids)
        rows: List[Dict[str, Any]] = []
        for vid in ids:
            p, v = per_video[str(vid)]["pred"], videos[vid]
            rows += [
                {"video_id": str(vid), "t_idx": j, "elapsed_min": float(v["elapsed"][j]),
                 "rsd_true": float(v["rsd"][j]), "rsd_pred": float(p[j])}
                for j in range(len(p))
            ]
        return {"metrics": compute_dataset_metrics(per_video, windows=protocol_windows(self.configs)), "rows": rows}
