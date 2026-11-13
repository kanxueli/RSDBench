"""BD-Net training (Wu et al., MICCAI 2022)."""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

from data_provider.datasets.catanet_frames import list_split_folders
from exp.exp_catanet import ExpCataNet, _link_split
from utils.io import ensure_dir, save_json, save_predictions_csv, save_summary
from utils.metrics import compute_dataset_metrics, protocol_windows
from utils.seed import set_seed


def _members(net):
    return [net.rnn1, net.rnn2, net.rnn3, net.rnn4, net.rnn5]


class ExpBDNet(ExpCataNet):
    def _bd_cfg(self) -> Dict[str, Any]:
        m = self.configs.get("model") or {}
        b = self.configs.get("bdnet") or m.get("bdnet") or {}
        return {
            "blstm_epochs": int(b.get("blstm_epochs", 80)),
            "blstm_lr": float(b.get("blstm_lr", 8e-4)),
            "ncl_lambda": float(b.get("ncl_lambda", 1e-3)),
            "gamma": float(b.get("gamma", 0.3)),
            "patience": int(b.get("patience", 15)),
            "reuse_catanet_cnn": bool(b.get("reuse_catanet_cnn", False)),
            "catanet_exp": str(b.get("catanet_exp", "cataract101_CataNet")),
            "use_experience": bool(m.get("use_experience", True)),
        }

    def _pool_and_layout(self, seed: int, train_ids, val_ids) -> Path:
        # Own layout dir: CataNet relinks layout/seed{N} and may run concurrently.
        root = Path(self.repo_root)
        pool = self._ensure_pool(list(train_ids) + list(val_ids or []))
        layout = root / ".cache" / f"catanet_{self._dataset()}" / "layout_bdnet" / f"seed{seed}"
        _link_split(train_ids, pool, layout / "train")
        _link_split(val_ids or [], pool, layout / "val")
        self._pool = pool
        return layout

    # ---- stage 1 -------------------------------------------------------
    def _stage1(self, model, layout: Path, rid_dir: Path, seed: int, st, bd) -> Path:
        out_cnn = rid_dir / "cnn"
        ensure_dir(out_cnn)
        if bd["reuse_catanet_cnn"] and not (out_cnn / "DONE").exists():
            src = Path(self.configs.get("output_dir", "results")) / bd["catanet_exp"] / f"seed{seed}" / "cnn"
            if (src / "DONE").exists() and (src / "catRSDNet_CNN.pth").exists():
                shutil.copy2(src / "catRSDNet_CNN.pth", out_cnn / "catRSDNet_CNN.pth")
                (out_cnn / "DONE").write_text(f"copied from {src}\n", encoding="utf-8")
                print(f"[BDNet] stage1 CNN reused from {src}", flush=True)
        return self._train_cnn(model, layout, out_cnn, st)

    # ---- features ------------------------------------------------------
    def _split_data(self, model, layout: Path, split: str, cache: Path, st) -> Dict[str, Dict]:
        vdirs, labels = list_split_folders(layout, split)
        vdirs = [d if d.endswith(os.sep) else d + os.sep for d in vdirs]
        img_transform = Compose([ToPILImage(), Resize(st["input_size"]), ToTensor()])
        feats = self._cached_features(model, cache, list(zip(labels, vdirs)), img_transform, st)
        out = {}
        for lab, vd in zip(labels, vdirs):
            y = np.genfromtxt(lab, delimiter=",", skip_header=1)[:, 1:]
            x = feats[vd]
            if len(x) != len(y):
                raise ValueError(f"{vd}: {len(x)} frames but {len(y)} label rows in {lab}")
            if Path(lab).stem != Path(vd.rstrip(os.sep)).name:
                raise ValueError(f"label file {lab} does not belong to video dir {vd}")
            out[vd] = {"x": x, "y": y}
        return out

    # ---- stage 2 -------------------------------------------------------
    def _bd_loss(self, outs, y_rsd, y_step, y_exp, bd):
        rsd_all = torch.stack([o[2].reshape(-1) for o in outs], dim=0)  # (K,T)
        rsd_bar = rsd_all.mean(dim=0).detach()
        total = 0.0
        for step, exp, rsd, s in outs:
            rsd = rsd.reshape(-1)
            s = s.reshape(-1).clamp(-10.0, 10.0)  # s = log σ²; clamp only guards exp overflow
            l_rsd = (0.5 * torch.exp(-s) * (rsd - y_rsd) ** 2 + 0.5 * s).mean()
            l_ncl = -((rsd - rsd_bar) ** 2).mean()
            l_aux = F.cross_entropy(step, y_step)
            if bd["use_experience"]:
                l_aux = l_aux + F.cross_entropy(exp, y_exp)
            total = total + l_rsd + bd["ncl_lambda"] * l_ncl + bd["gamma"] * l_aux
        return total

    @torch.no_grad()
    def _predict_full(self, net, x: torch.Tensor) -> np.ndarray:
        """Whole sequence in one call (one dropout mask per sequence) — used for val."""
        rsd = torch.stack([r(x, stateful=False)[2].reshape(-1) for r in _members(net)], dim=0)
        return rsd.mean(dim=0).cpu().numpy()

    @torch.no_grad()
    def _predict_stateful(self, net, x: torch.Tensor) -> np.ndarray:
        preds = []
        for i in range(x.size(0)):
            xi = x[i : i + 1]
            r = [m(xi, stateful=i > 0)[2].reshape(-1) for m in _members(net)]
            preds.append(torch.stack(r, dim=0).mean(dim=0))
        return torch.cat(preds).cpu().numpy()

    def _metrics(self, net, data: Dict[str, Dict], stateful: bool):
        net.eval()
        per_video, rows = {}, []
        for vd, d in data.items():
            vid = Path(vd.rstrip(os.sep)).name
            x = torch.from_numpy(d["x"]).float().to(self.device).unsqueeze(1)
            pred = self._predict_stateful(net, x) if stateful else self._predict_full(net, x)
            tgt = d["y"][:, 5] / 60.0 / 25.0
            per_video[vid] = {"pred": pred, "target": tgt}
            for j in range(len(pred)):
                rows.append(
                    {
                        "video_id": vid,
                        "t_idx": j,
                        "elapsed_min": j / (2.5 * 60.0),
                        "rsd_true": float(tgt[j]),
                        "rsd_pred": float(pred[j]),
                    }
                )
        return compute_dataset_metrics(per_video, windows=protocol_windows(self.configs)), rows

    def _stage2(self, model, train_data, val_data, rid_dir: Path, bd) -> None:
        net = model.net
        out = rid_dir / "blstm"
        ensure_dir(out)
        best_p, last_p = out / "best.pt", out / "last.pt"
        params = [p for r in _members(net) for p in r.parameters()]
        net.freeze_cnn(True)
        net.freeze_rnn(False)
        optim = torch.optim.Adam(params, lr=bd["blstm_lr"])
        start, best, bad, history = 0, float("inf"), 0, []
        if last_p.exists():
            st = torch.load(last_p, map_location=self.device)
            net.load_state_dict(st["model_dict"])
            optim.load_state_dict(st["optim"])
            start, best, bad, history = int(st["epoch"]), float(st["best"]), int(st["bad"]), list(st["history"])
            print(f"[resume] BDNet stage2 epoch={start} best={best:.4f} bad={bad}", flush=True)
            if st.get("stopped"):
                start = bd["blstm_epochs"]

        keys = list(train_data.keys())
        for ep in range(start, bd["blstm_epochs"]):
            net.train()
            random.shuffle(keys)
            tot = 0.0
            for vd in keys:
                d = train_data[vd]
                x = torch.from_numpy(d["x"]).float().to(self.device).unsqueeze(1)  # (T,1,1664)
                y = torch.from_numpy(d["y"]).float().to(self.device)
                outs = [r(x, stateful=False) for r in _members(net)]
                loss = self._bd_loss(outs, y[:, 5] / 60.0 / 25.0, y[:, 0].long(), (y[:, 2] - 1).long(), bd)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                tot += float(loss.item())
            rec = {"epoch": ep + 1, "train_loss": tot / max(len(keys), 1)}
            stop = False
            if val_data:
                vm, _ = self._metrics(net, val_data, stateful=False)
                mae = float(vm["mae_all"])
                rec["val_mae_all"] = mae
                improved = mae < best
                if improved:
                    best, bad = mae, 0
                    torch.save({"epoch": ep + 1, "model_dict": net.state_dict()}, best_p)
                else:
                    bad += 1
                print(
                    f"[blstm] epoch={ep+1}/{bd['blstm_epochs']} train={rec['train_loss']:.4f} "
                    f"val_mae={mae:.4f} best={best:.4f} bad={bad}/{bd['patience']}"
                    + ("  **" if improved else ""),
                    flush=True,
                )
                if bd["patience"] > 0 and bad >= bd["patience"]:
                    print(f"[early_stop] BDNet stage2 epoch={ep+1} best={best:.4f}", flush=True)
                    stop = True
            else:
                print(f"[blstm] epoch={ep+1} train={rec['train_loss']:.4f}", flush=True)
            history.append(rec)
            torch.save(
                {
                    "epoch": ep + 1,
                    "model_dict": net.state_dict(),
                    "optim": optim.state_dict(),
                    "best": best,
                    "bad": bad,
                    "history": history,
                    "stopped": stop,
                },
                last_p,
            )
            if stop:
                break
        if best_p.exists():
            net.load_state_dict(torch.load(best_p, map_location=self.device)["model_dict"])
            print(f"[best] BDNet loaded {best_p} val_mae={best:.4f}", flush=True)
        save_json({"history": history, "best_val_mae": best}, out / "history.json")

    def run(self) -> Dict[str, Any]:
        scheme = self.splits.get("scheme")
        if scheme not in ("repeated_holdout", "stratified_repeated_holdout"):
            raise ValueError(f"BDNet runner expects stratified_repeated_holdout, got {scheme}")
        st = self._stage_cfg()
        bd = self._bd_cfg()
        repeat_metrics = {}
        for rep in self.splits["repeats"]:
            seed = int(rep["seed"])
            rid = f"seed{seed}"
            rid_dir = self.output_dir / rid
            if (rid_dir / "metrics.json").exists():
                print(f"[skip] {rid} already has metrics.json", flush=True)
                repeat_metrics[rid] = json.loads((rid_dir / "metrics.json").read_text())
                continue
            set_seed(seed)
            self.configs["_current_seed"] = seed
            ensure_dir(rid_dir)
            model = self._build_model()
            layout = self._pool_and_layout(seed, rep["train"], rep.get("val"))
            _link_split(rep["test"], self._ensure_pool(rep["test"]), layout / "test")
            print(f"[BDNet] layout={layout} bd={bd}", flush=True)

            cnn_ckpt = self._stage1(model, layout, rid_dir, seed, st, bd)
            model.net.cnn.load_state_dict(torch.load(cnn_ckpt, map_location=self.device)["model_dict"])
            model.net.set_cnn_as_feature_extractor()
            model = model.to(self.device)

            cache = rid_dir / "feats"
            data = {s: self._split_data(model, layout, s, cache, st) for s in ("train", "val", "test")}
            self._stage2(model, data["train"], data["val"], rid_dir, bd)

            metrics, rows = self._metrics(model.net, data["test"], stateful=True)
            torch.save(model.net.state_dict(), rid_dir / "model.pth")
            save_predictions_csv(rows, rid_dir / "predictions.csv")
            save_json(metrics, rid_dir / "metrics.json")
            repeat_metrics[rid] = metrics
            print(f"[BDNet] {rid} test mae_all={metrics['mae_all']:.4f}", flush=True)
            shutil.rmtree(cache, ignore_errors=True)
        return save_summary(self.output_dir, repeat_metrics)
