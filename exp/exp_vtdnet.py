"""VTD-Net training (Bodenstedt et al., IJCARS 2019)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from data_provider.datasets.feature_rsd import FeatureRSD
from exp.exp_rsd import ExpRSD
from utils.io import ensure_dir, save_json
from utils.metrics import compute_dataset_metrics, protocol_windows


class ExpVTDNet(ExpRSD):
    def _feature_dir(self) -> Path:
        tmpl = self.configs.get("feature_dir")
        if not tmpl:
            raise ValueError("feature_dir required (e.g. .cache/features/<ds>/VTD_alexnetSiamese_seed{seed})")
        return Path(str(tmpl).format(seed=int(self.configs.get("seed", 42))))

    def _loader(self, ids, shuffle: bool):
        return torch.utils.data.DataLoader(
            FeatureRSD(self._feature_dir(), ids), batch_size=1, shuffle=shuffle, num_workers=0,
            collate_fn=lambda xs: xs[0],
        )

    def _to_dev(self, batch):
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def _train_one(self, model, train_ids, val_ids=None):
        es = self._es_cfg()
        epochs = int(self.configs.get("epochs", 100))
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                               lr=float(self.configs.get("lr", 1e-6)),
                               weight_decay=float(self.configs.get("weight_decay", 0.0)))
        use_val = bool(val_ids) and es["enabled"]
        ckpt_dir = Path(getattr(self, "_ckpt_dir", None) or self.output_dir / "ckpt")
        ensure_dir(ckpt_dir)
        last_p, best_p = ckpt_dir / "last.pt", ckpt_dir / "best.pt"

        start_ep, best, bad, history = 0, float("inf"), 0, []
        if last_p.exists() and bool(self.configs.get("resume", True)):
            st = torch.load(last_p, map_location=self.device)
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["optim"])
            start_ep, best, bad = int(st["epoch"]), float(st["best"]), int(st["bad"])
            history = list(st.get("history", []))
            print(f"[resume] {last_p} epoch={start_ep} best={best:.4f} bad={bad}", flush=True)
            if st.get("stopped", False):
                start_ep = epochs

        loss_kind = str(self.configs.get("vtd_loss", "l1_rsd"))
        grad_clip = float(self.configs.get("grad_clip", 0.0) or 0.0)
        loader = self._loader(list(train_ids), shuffle=True)
        for ep in range(start_ep, epochs):
            model.train()
            total, n = 0.0, 0
            for batch in loader:
                batch = self._to_dev(batch)
                out = model(batch)
                if loss_kind == "bce_progress":
                    y = out["progress"]
                    label = torch.arange(1, y.numel() + 1, device=y.device, dtype=y.dtype) / y.numel()
                    loss = F.binary_cross_entropy(y.clamp(1e-6, 1 - 1e-6), label)
                elif loss_kind == "l1_rsd":
                    loss = F.l1_loss(out["rsd"], batch["rsd"].float().reshape(-1))
                else:
                    raise ValueError(f"vtd_loss={loss_kind!r}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                total += float(loss.item())
                n += 1
            rec = {"epoch": ep + 1, "train_loss": total / max(n, 1)}
            stop = False
            msg = f"[train] epoch={ep+1}/{epochs} loss={rec['train_loss']:.5f}"
            if use_val and ((ep + 1) % max(es["val_every"], 1) == 0 or ep + 1 == epochs):
                score = float(self._eval(model, val_ids, split="val")["metrics"][es["monitor"]])
                rec["val_" + es["monitor"]] = score
                if score < best:
                    best, bad = score, 0
                    torch.save(model.state_dict(), best_p)
                    msg += f" val_{es['monitor']}={score:.4f}  **"
                else:
                    bad += 1
                    msg += f" val_{es['monitor']}={score:.4f} bad={bad}/{es['patience']}"
                if es["patience"] > 0 and bad >= es["patience"] and ep + 1 >= es["min_epochs"]:
                    stop = True
            print(msg, flush=True)
            history.append(rec)
            torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "epoch": ep + 1, "best": best,
                        "bad": bad, "history": history, "stopped": stop}, last_p)
            if stop:
                print(f"[early_stop] epoch={ep+1} best={best:.4f}", flush=True)
                break

        if use_val and best_p.exists():
            model.load_state_dict(torch.load(best_p, map_location=self.device))
            print(f"[best] loaded {best_p} ({es['monitor']}={best:.4f})", flush=True)
        save_json({"history": history, "best": best, "monitor": es["monitor"]}, ckpt_dir / "history.json")
        return model

    @torch.no_grad()
    def _eval(self, model, video_ids, split: str = "test"):
        model.eval()
        per_video: Dict[str, Dict[str, Any]] = {}
        rows: List[Dict[str, Any]] = []
        for batch in self._loader(list(video_ids), shuffle=False):
            batch = self._to_dev(batch)
            pred = model(batch)["rsd"].cpu().numpy().reshape(-1)
            tgt = batch["rsd"].cpu().numpy().reshape(-1)
            el = batch["elapsed"].cpu().numpy().reshape(-1)
            vid = str(batch["video_id"])
            per_video[vid] = {"pred": pred, "target": tgt}
            rows += [{"video_id": vid, "t_idx": j, "elapsed_min": float(el[j]), "rsd_true": float(tgt[j]),
                      "rsd_pred": float(pred[j])} for j in range(len(pred))]
        return {"metrics": compute_dataset_metrics(per_video, windows=protocol_windows(self.configs)), "rows": rows}
