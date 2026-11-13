"""PitRSDNet training (Wijekoon et al., HTL 2024)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomResizedCrop, Resize, ToPILImage, ToTensor
from tqdm import tqdm

from data_provider.data_factory import _data_root
from data_provider.datasets.base_rsd import elapsed_and_rsd_minutes
from exp.exp_rsd import ExpRSD
from layers.time_channel import append_elapsed_channel
from preprocessing.finetune_cnn import build_video_plans
from preprocessing.prepare_frames_1fps import DEFAULT_OUT
from utils.io import ensure_dir
from utils.metrics import compute_dataset_metrics, protocol_windows

SAMPLE_FPS = 1.0


class _Frames(Dataset):
    """All 1fps frames of a video list → (RGB tensor, elapsed minutes, step label)."""

    def __init__(self, videos: Dict[Any, Dict[str, Any]], ids, transform):
        self.videos = videos
        self.transform = transform
        self.items = [(vid, k) for vid in ids for k in range(len(videos[vid]["phase"]))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        vid, k = self.items[i]
        v = self.videos[vid]
        img = self.transform(np.asarray(np.load(v["frames"], mmap_mode="r")[k]))
        return img, torch.tensor(float(v["elapsed"][k])), torch.tensor(int(v["phase"][k]))


class ExpPitRSDNet(ExpRSD):
    def _st(self) -> Dict[str, Any]:
        s = {k: yaml.safe_load(v) if isinstance(v, str) else v for k, v in dict(self.configs.get("stages") or {}).items()}
        return {
            "cnn_epochs": int(s.get("cnn_epochs", 20)),
            "cnn_patience": int(s.get("cnn_patience", 10)),
            "cnn_lr": float(s.get("cnn_lr", 1e-4)),
            "cnn_batch_size": int(s.get("cnn_batch_size", 50)),
            "val_batch_size": int(s.get("val_batch_size", 150)),
            "sub_epoch_validation": int(s.get("sub_epoch_validation", 100)),
            "lstm_epochs": list(s.get("lstm_epochs", [40, 40])),
            "lstm_lrs": list(s.get("lstm_lrs", [1e-3, 1e-4])),
            "lstm_patience": int(s.get("lstm_patience", 15)),
            "input_size": int(s.get("input_size", 224)),
            "feature_batch": int(s.get("feature_batch", 150)),
            "num_workers": int(s.get("num_workers", 4)),
        }

    def _videos(self, ids) -> Dict[Any, Dict[str, Any]]:
        ds = self.configs["dataset"]
        frame_dir = Path(self.configs.get("frame_cache_1fps") or DEFAULT_OUT) / ds
        plans = build_video_plans(ds, _data_root(self.configs, ds), ids, sample_fps=SAMPLE_FPS)
        out = {}
        for vid, (_, idx, phase, _) in plans.items():
            path = frame_dir / f"{vid}.npy"
            if not path.exists():
                raise FileNotFoundError(f"{path} — run preprocessing/prepare_frames_1fps.py --dataset {ds}")
            elapsed, rsd, _ = elapsed_and_rsd_minutes(len(idx), SAMPLE_FPS)
            out[vid] = {
                "frames": str(path),
                "phase": np.asarray(phase, dtype=np.int64),
                "elapsed": elapsed.astype(np.float32),
                "rsd": rsd.astype(np.float32),
            }
        return out

    def _class_weights(self, videos, train_ids, n_cls) -> torch.Tensor:
        counts = np.zeros(n_cls)
        for vid in train_ids:
            counts += np.bincount(videos[vid]["phase"], minlength=n_cls)[:n_cls]
        w = np.zeros(n_cls)
        w[counts > 0] = 1.0 / counts[counts > 0]
        return torch.tensor(w / w.max(), dtype=torch.float32, device=self.device)

    def _with_time(self, model, img, elapsed):
        return append_elapsed_channel(img, elapsed, model.max_len)

    def _finetune_cnn(self, model, videos, train_ids, val_ids, out_dir: Path, st, cw):
        size = st["input_size"]
        tf_train = Compose(
            [ToPILImage(), RandomHorizontalFlip(), RandomResizedCrop(size, scale=(0.4, 1.0), ratio=(1.0, 1.0)), ToTensor()]
        )
        tf_val = Compose([ToPILImage(), Resize(size), ToTensor()])
        nw = st["num_workers"]
        train_loader = DataLoader(
            _Frames(videos, train_ids, tf_train), batch_size=st["cnn_batch_size"], shuffle=True,
            num_workers=nw, pin_memory=True, drop_last=False,
        )
        val_loader = (
            DataLoader(_Frames(videos, val_ids, tf_val), batch_size=st["val_batch_size"], shuffle=False,
                       num_workers=nw, pin_memory=True)
            if val_ids else None
        )
        head = nn.Linear(model.feature_size, model.n_phases).to(self.device)
        params = list(model.cnn.parameters()) + list(head.parameters())
        optim = torch.optim.Adam(params, lr=st["cnn_lr"])
        criterion = nn.CrossEntropyLoss(weight=cw)
        ckpt = out_dir / "cnn_step_ft.pth"
        done = out_dir / "cnn_DONE"
        if done.exists() and ckpt.exists():
            print(f"[resume] stage-1 done → {ckpt}", flush=True)
            model.cnn.load_state_dict(torch.load(ckpt, map_location=self.device)["cnn"])
            return
        best = float("inf")
        cnn_patience = int(st.get("cnn_patience", 0))
        bad = 0

        def step_logits(img, elapsed):
            return head(model.cnn(self._with_time(model, img.to(self.device), elapsed.to(self.device))))

        for ep in range(st["cnn_epochs"]):
            model.cnn.train()
            head.train()
            losses = []
            improved = False
            for ii, (img, elapsed, y) in enumerate(tqdm(train_loader, desc=f"[pit] cnn ep{ep+1}", leave=False)):
                loss = criterion(step_logits(img, elapsed), y.to(self.device))
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                losses.append(float(loss.item()))
                if val_loader is not None and ii % st["sub_epoch_validation"] == 0:
                    model.cnn.eval()
                    head.eval()
                    with torch.no_grad():
                        vl = [float(criterion(step_logits(i, e), t.to(self.device)).item()) for i, e, t in val_loader]
                    v = float(np.mean(vl))
                    if v < best:
                        best = v
                        improved = True
                        torch.save({"epoch": ep + 1, "cnn": model.cnn.state_dict(), "head": head.state_dict()}, ckpt)
                        print(f"  [pit/cnn] ep{ep+1} it{ii} val_loss={v:.4f} ** saved", flush=True)
                    model.cnn.train()
                    head.train()
            print(f"[pit/cnn] epoch={ep+1}/{st['cnn_epochs']} train_loss={np.mean(losses):.4f}", flush=True)
            if cnn_patience > 0 and val_loader is not None:
                if improved:
                    bad = 0
                else:
                    bad += 1
                if bad >= cnn_patience:
                    print(f"[early_stop] pit/cnn epoch={ep+1} best_val={best:.4f}", flush=True)
                    break
        if not ckpt.exists():
            torch.save({"epoch": st["cnn_epochs"], "cnn": model.cnn.state_dict(), "head": head.state_dict()}, ckpt)
        done.write_text("ok\n", encoding="utf-8")
        model.cnn.load_state_dict(torch.load(ckpt, map_location=self.device)["cnn"])

    @torch.no_grad()
    def _extract(self, model, videos, ids, st) -> Dict[Any, torch.Tensor]:
        model.cnn.eval()
        tf = Compose([ToPILImage(), Resize(st["input_size"]), ToTensor()])
        feats = {}
        for vid in tqdm(ids, desc="[pit] extract", leave=False):
            loader = DataLoader(_Frames(videos, [vid], tf), batch_size=st["feature_batch"], shuffle=False,
                                num_workers=st["num_workers"], pin_memory=True)
            chunks = [model.cnn(self._with_time(model, i.to(self.device), e.to(self.device))).cpu() for i, e, _ in loader]
            feats[vid] = torch.cat(chunks)
        return feats

    def _seq_loss(self, model, out, v, cw):
        dev = self.device
        phase = torch.from_numpy(v["phase"]).to(dev)
        rsd = torch.from_numpy(v["rsd"]).to(dev)
        loss_step = F.cross_entropy(out["phase_logits"], phase, weight=cw)
        loss_rsd = F.smooth_l1_loss(out["rsd"] / model.rsd_norm, rsd / model.rsd_norm)
        return loss_step + loss_rsd

    @torch.no_grad()
    def _val_scores(self, model, videos, feats, val_ids, cw):
        """(val MAE-ALL minutes, val loss) over full validation videos."""
        model.eval()
        per_video, losses = {}, []
        for v in val_ids:
            out = model.forward_features(feats[v].to(self.device))
            losses.append(float(self._seq_loss(model, out, videos[v], cw).item()))
            per_video[str(v)] = {"pred": out["rsd"].cpu().numpy().reshape(-1), "target": videos[v]["rsd"]}
        mae = float(compute_dataset_metrics(per_video, windows=protocol_windows(self.configs))["mae_all"])
        return mae, float(np.mean(losses))

    def _train_lstm(self, model, videos, feats, train_ids, val_ids, out_dir: Path, st, cw):
        for p in model.cnn.parameters():
            p.requires_grad_(False)
        params = [p for n, p in model.named_parameters() if not n.startswith("cnn.")]
        optim = torch.optim.Adam(params, lr=st["lstm_lrs"][0])
        ckpt, last_p = out_dir / "lstm_best.pth", out_dir / "lstm_last.pt"
        head_state = lambda: {k: t for k, t in model.state_dict().items() if not k.startswith("cnn.")}
        patience = int(st["lstm_patience"])
        phase, ep_in_phase, ep_total, best, bad, history, done = 0, 0, 0, float("inf"), 0, [], False
        if last_p.exists():
            r = torch.load(last_p, map_location=self.device)
            model.load_state_dict(r["model"], strict=False)
            optim.load_state_dict(r["optim"])
            phase, ep_in_phase, ep_total = int(r["phase"]), int(r["ep_in_phase"]), int(r["ep_total"])
            best, bad, history, done = float(r["best"]), int(r["bad"]), list(r["history"]), bool(r["done"])
            print(f"[resume] stage-2 phase={phase} ep_in_phase={ep_in_phase} best={best:.4f}", flush=True)

        while not done and phase < len(st["lstm_epochs"]):
            lr = float(st["lstm_lrs"][phase])
            for g in optim.param_groups:
                g["lr"] = lr
            if ep_in_phase >= int(st["lstm_epochs"][phase]):
                phase, ep_in_phase, bad = phase + 1, 0, 0
                continue
            ep_in_phase += 1
            ep_total += 1
            model.train()
            order = list(train_ids)
            random.shuffle(order)
            tr = []
            for vid in order:
                out = model.forward_features(feats[vid].to(self.device))
                loss = self._seq_loss(model, out, videos[vid], cw)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                tr.append(float(loss.item()))
            mae, vloss = self._val_scores(model, videos, feats, val_ids, cw) if val_ids else (float(np.mean(tr)), float(np.mean(tr)))
            mark = ""
            if mae < best:
                best, bad, mark = mae, 0, "  **"
                torch.save(head_state(), ckpt)
            else:
                bad += 1
            history.append({"epoch": ep_total, "phase": phase, "lr": lr, "train": float(np.mean(tr)), "val_loss": vloss, "val_mae": mae})
            print(
                f"[pit/lstm] epoch={ep_total} phase={phase + 1} lr={lr:g} train={np.mean(tr):.4f} "
                f"val_loss={vloss:.4f} val_mae={mae:.4f} best={best:.4f} bad={bad}/{patience}{mark}",
                flush=True,
            )
            if patience > 0 and bad >= patience:
                if phase + 1 < len(st["lstm_epochs"]):
                    print(f"[early_stop] phase {phase + 1} at epoch {ep_total} → next lr phase from best", flush=True)
                    model.load_state_dict(torch.load(ckpt, map_location=self.device), strict=False)
                    phase, ep_in_phase, bad = phase + 1, 0, 0
                else:
                    print(f"[early_stop] stage-2 at epoch {ep_total} best val_mae={best:.4f}", flush=True)
                    done = True
            torch.save(
                {"model": head_state(), "optim": optim.state_dict(), "phase": phase, "ep_in_phase": ep_in_phase,
                 "ep_total": ep_total, "best": best, "bad": bad, "history": history, "done": done},
                last_p,
            )
        model.load_state_dict(torch.load(ckpt, map_location=self.device), strict=False)
        print(f"[best] stage-2 loaded {ckpt} val_mae={best:.4f}", flush=True)
        (out_dir / "lstm_history.json").write_text(__import__("json").dumps(history, indent=1), encoding="utf-8")

    def _train_one(self, model, train_ids, val_ids=None):
        st = self._st()
        seed = int(self.configs.get("seed", 42))
        out_dir = ensure_dir(self.output_dir / f"seed{seed}")
        val_ids = list(val_ids or [])
        self._videos_cache = self._videos(list(train_ids) + val_ids)
        cw = self._class_weights(self._videos_cache, train_ids, model.n_phases)
        print(f"[pit] seed={seed} train={len(train_ids)} val={len(val_ids)} stages={st}", flush=True)
        self._finetune_cnn(model, self._videos_cache, train_ids, val_ids, out_dir, st, cw)
        feats = self._extract(model, self._videos_cache, list(train_ids) + val_ids, st)
        self._train_lstm(model, self._videos_cache, feats, train_ids, val_ids, out_dir, st, cw)
        return model

    @torch.no_grad()
    def _eval(self, model, video_ids):
        model.eval()
        st = self._st()
        videos = self._videos(video_ids)
        feats = self._extract(model, videos, list(video_ids), st)
        per_video, rows = {}, []
        for vid in video_ids:
            pred = model.forward_features(feats[vid].to(self.device))["rsd"].cpu().numpy().reshape(-1)
            v = videos[vid]
            per_video[str(vid)] = {"pred": pred, "target": v["rsd"]}
            rows += [
                {"video_id": str(vid), "t_idx": j, "elapsed_min": float(v["elapsed"][j]),
                 "rsd_true": float(v["rsd"][j]), "rsd_pred": float(pred[j])}
                for j in range(len(pred))
            ]
        return {"metrics": compute_dataset_metrics(per_video, windows=protocol_windows(self.configs)), "rows": rows}
