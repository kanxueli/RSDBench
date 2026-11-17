"""SUPR-GAN training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from exp.exp_basic import ExpBasic
from exp.exp_rivoir_common import gpu_window, load_videos, phase_refs_from_videos, read_frames
from utils.io import ensure_dir, save_json, save_predictions_csv, save_summary
from utils.metrics import compute_dataset_metrics, protocol_windows
from utils.seed import set_seed
from utils.splits import build_protocol_splits, save_splits

class ExpSUPRGAN(ExpBasic):
    def __init__(self, configs, repo_root):
        super().__init__(configs)
        self.repo_root = repo_root
        self.splits = build_protocol_splits(configs, repo_root)
        ensure_dir(self.output_dir)
        save_splits(self.splits, self.output_dir / "splits.json")
        self.model = self._build_model()
        self._phase_w: Optional[torch.Tensor] = None

    def _cfg(self):
        m = self.configs.get("model") or {}
        return {
            "lr": float(self.configs.get("lr", 1e-4)),
            "pretrain_epochs": int(m.get("pretrain_epochs", 60)),
            "phase_only_epochs": int(m.get("phase_only_epochs", 0)),
            "pretrain_win": int(m.get("pretrain_win", 128)),
            "gan_epochs": int(self.configs.get("epochs", 2000)),
            "epoch_size": int(m.get("epoch_size", 64)),
            "tp": int(m.get("past_horizon", 15)),
            "tf": int(m.get("future_horizon", 15)),
            "ns": int(m.get("n_samples", 10)),
            "w_dis": float(m.get("w_dis", 0.6)),
            "w_rec": float(m.get("w_rec", 0.2)),
            "w_past": float(m.get("w_past", 1.0)),
            "w_rsd": float(m.get("w_rsd", 1.5)),
            "w_prog": float(m.get("w_prog", 0.1)),
            "end_loss_gamma": float(m.get("end_loss_gamma", 2.0)),
            "label_smooth": float(m.get("label_smooth", 0.05)),
            "patience": int((self.configs.get("early_stopping") or {}).get("patience", 30)),
            "val_every": int((self.configs.get("early_stopping") or {}).get("val_every", 50)),
            "gan_batch": int(m.get("gan_batch", 8)),
            "select_on": str(m.get("select_on", "mae_5")),
        }

    def _class_weights(self, videos, train_ids, n_cls) -> torch.Tensor:
        counts = np.zeros(n_cls, dtype=np.float64)
        for vid in train_ids:
            counts += np.bincount(videos[vid]["phase"], minlength=n_cls)[:n_cls]
        w = np.zeros(n_cls, dtype=np.float64)
        mask = counts > 0
        w[mask] = 1.0 / counts[mask]
        w = w / max(w.max(), 1e-8)
        return torch.tensor(w, dtype=torch.float32, device=self.device)

    def _phase_ce(self, logits, tgt):
        cfg = self._cfg()
        return F.cross_entropy(
            logits,
            tgt,
            weight=self._phase_w,
            label_smoothing=cfg["label_smooth"],
        )

    def _rsd_loss(self, pred, tgt, gamma: float):
        if gamma > 0:
            w = 1.0 + gamma / (tgt.detach().abs() + 0.5)
            return (w * F.smooth_l1_loss(pred, tgt, reduction="none")).mean()
        return F.smooth_l1_loss(pred, tgt)

    def _prog_loss(self, pred_prog, elapsed, rsd):
        tgt = (elapsed / (elapsed + rsd + 1e-3)).clamp(0.0, 1.0)
        return F.binary_cross_entropy(pred_prog, tgt)

    def _sample_clip(self, videos, ids, tp, tf):
        vid = ids[int(np.random.randint(0, len(ids)))]
        v = videos[vid]
        T = len(v["phase"])
        need = tp + tf
        if T <= need:
            t0 = tp - 1
            frames = read_frames(v["frames"])  # CPU
            phase = torch.from_numpy(v["phase"]).to(self.device)
            past_idx = [min(max(0, t0 - tp + 1 + i), T - 1) for i in range(tp)]
            fut_idx = [min(max(0, t0 + 1 + i), T - 1) for i in range(tf)]
        else:
            t0 = int(np.random.randint(tp - 1, T - tf))
            frames = read_frames(v["frames"])  # CPU
            phase = torch.from_numpy(v["phase"]).to(self.device)
            past_idx = list(range(t0 - tp + 1, t0 + 1))
            fut_idx = list(range(t0 + 1, t0 + 1 + tf))
        past_f = frames[past_idx].to(self.device, non_blocking=True)
        past_ph = phase[past_idx]
        fut_ph = phase[fut_idx]
        elapsed = torch.from_numpy(v["elapsed"]).to(self.device)
        rsd = torch.from_numpy(v["rsd"]).to(self.device)
        return past_f, past_ph, fut_ph, elapsed[past_idx], rsd[past_idx]

    def _pretrain_encoder(self, model, videos, train_ids, cfg, ckpt: Path):
        done = ckpt / "encoder_DONE"
        path = ckpt / "encoder.pth"
        if done.exists() and path.exists():
            model.load_state_dict(torch.load(path, map_location=self.device), strict=False)
            print(f"[resume] encoder pretrain → {path}", flush=True)
            return
        opt = torch.optim.Adam(
            list(model.cnn.parameters())
            + list(model.enc_lstm.parameters())
            + list(model.phase_head.parameters())
            + list(model.prog_head.parameters())
            + list(model.rsd_head.parameters()),
            lr=cfg["lr"],
        )
        ids = list(train_ids)
        win_cap = max(16, cfg["pretrain_win"])
        phase_only = max(0, cfg["phase_only_epochs"])
        for ep in range(cfg["pretrain_epochs"]):
            model.train()
            np.random.shuffle(ids)
            total, n, correct, count = 0.0, 0, 0, 0
            only_phase = ep < phase_only
            for vid in tqdm(ids, desc=f"pretrain {ep+1}/{cfg['pretrain_epochs']}", leave=False):
                frames = read_frames(videos[vid]["frames"])  # CPU
                phase = torch.from_numpy(videos[vid]["phase"]).to(self.device)
                elapsed = torch.from_numpy(videos[vid]["elapsed"]).to(self.device)
                rsd = torch.from_numpy(videos[vid]["rsd"]).to(self.device)
                T = frames.size(0)
                win = min(win_cap, T)
                s = 0 if T == win else int(np.random.randint(0, T - win + 1))
                el_w = elapsed[s : s + win]
                enc = model.encode_video(gpu_window(frames, s, s + win, self.device), el_w)
                logits = enc["phase_logits"]
                tgt = phase[s : s + win]
                loss = self._phase_ce(logits, tgt)
                if not only_phase:
                    rsd_w = rsd[s : s + win]
                    pred_rsd = model.predict_rsd(enc["h"], logits, el_w)
                    loss = (
                        loss
                        + cfg["w_rsd"] * self._rsd_loss(pred_rsd, rsd_w, cfg["end_loss_gamma"])
                        + cfg["w_prog"] * self._prog_loss(enc["progress"], el_w, rsd_w)
                    )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.item())
                n += 1
                correct += int((logits.argmax(-1) == tgt).sum().item())
                count += int(tgt.numel())
            acc = correct / max(count, 1)
            tag = "phase_only" if only_phase else "joint"
            print(
                f"[pretrain] epoch={ep+1} mode={tag} loss={total/max(n,1):.4f} phase_acc={acc:.3f}",
                flush=True,
            )
        torch.save(model.state_dict(), path)
        done.write_text("ok")

    def _gan_step(self, model, videos, ids, cfg, opt_g, opt_d):
        past_f, past_ph, fut_ph, past_el, past_rsd = self._sample_clip(
            videos, ids, cfg["tp"], cfg["tf"]
        )
        past_oh = F.one_hot(past_ph, model.n_phases).float()
        fut_oh = F.one_hot(fut_ph, model.n_phases).float()

        # --- Discriminator ---
        enc = model.encode_video(past_f, past_el)
        with torch.no_grad():
            fake, _ = model.decode_future(enc["h"][-1], n_samples=1)
            fake = fake.squeeze(0)  # (Tf,P)
        real_logit = model.discriminate(past_oh.unsqueeze(0), fut_oh.unsqueeze(0))
        fake_logit = model.discriminate(past_oh.unsqueeze(0), fake.unsqueeze(0))
        loss_d = (
            -torch.log(torch.sigmoid(real_logit) + 1e-8).mean()
            - torch.log(1 - torch.sigmoid(fake_logit) + 1e-8).mean()
        )
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()

        # --- Generator ---
        enc = model.encode_video(past_f, past_el)
        fakes, fake_logits = model.decode_future(enc["h"][-1], n_samples=cfg["ns"])  # (Ns,Tf,P)
        fut_exp = fut_ph.unsqueeze(0).expand(cfg["ns"], -1)
        ce = F.cross_entropy(
            fake_logits.reshape(-1, model.n_phases), fut_exp.reshape(-1), reduction="none"
        )
        ce = ce.view(cfg["ns"], cfg["tf"]).mean(-1)
        loss_rec = ce.min()
        adv = 0.0
        for j in range(cfg["ns"]):
            logit = model.discriminate(past_oh.unsqueeze(0), fakes[j].unsqueeze(0))
            adv = adv + (-torch.log(torch.sigmoid(logit) + 1e-8).mean())
        loss_dis = adv / cfg["ns"]
        loss_past = self._phase_ce(enc["phase_logits"], past_ph)
        pred_rsd = model.predict_rsd(enc["h"], enc["phase_logits"], past_el)
        loss_rsd = self._rsd_loss(pred_rsd, past_rsd, cfg["end_loss_gamma"])
        loss_prog = self._prog_loss(enc["progress"], past_el, past_rsd)
        loss_g = (
            cfg["w_dis"] * loss_dis
            + cfg["w_rec"] * loss_rec
            + cfg["w_past"] * loss_past
            + cfg["w_rsd"] * loss_rsd
            + cfg["w_prog"] * loss_prog
        )
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
        del past_f
        return float(loss_g.item()), float(loss_d.item())

    def _select_score(self, metrics: Dict[str, float], phase_acc: float, select_on: str):
        key = select_on.lower()
        if key == "phase_acc":
            return phase_acc, "phase_acc", True  # higher better
        if key not in metrics:
            key = "mae_all"
        return float(metrics[key]), key, False

    def _train_one(self, model, train_ids, val_ids=None):
        cfg = self._cfg()
        ids_all = list(dict.fromkeys(list(train_ids) + list(val_ids or [])))
        videos = load_videos(self.configs, ids_all)
        refs = phase_refs_from_videos(videos, list(train_ids), model.n_phases)
        model.set_phase_refs(refs)
        self._phase_w = self._class_weights(videos, list(train_ids), model.n_phases)
        ckpt = Path(getattr(self, "_ckpt_dir"))
        ensure_dir(ckpt)
        self._pretrain_encoder(model, videos, train_ids, cfg, ckpt)

        g_params = [p for n, p in model.named_parameters() if not n.startswith("d_")]
        d_params = [p for n, p in model.named_parameters() if n.startswith("d_")]
        opt_g = torch.optim.Adam(g_params, lr=cfg["lr"])
        opt_d = torch.optim.Adam(d_params, lr=cfg["lr"])
        best_p, last_p = ckpt / "best.pt", ckpt / "last.pt"
        higher_better = cfg["select_on"] == "phase_acc"
        start, best, bad = 0, (-1.0 if higher_better else float("inf")), 0
        if last_p.exists() and bool(self.configs.get("resume", True)):
            st = torch.load(last_p, map_location=self.device)
            model.load_state_dict(st["model"], strict=False)
            start, best, bad = int(st["epoch"]), float(st["best"]), int(st["bad"])
            print(f"[resume] GAN epoch={start} best={best:.4f}", flush=True)

        train_ids = list(train_ids)
        gb = max(1, cfg["gan_batch"])
        for ep in range(start, cfg["gan_epochs"]):
            model.train()
            gl, dl = 0.0, 0.0
            nsteps = max(1, cfg["epoch_size"] // gb)
            for _ in range(nsteps):
                g_acc = d_acc = 0.0
                for _b in range(gb):
                    g, d = self._gan_step(model, videos, train_ids, cfg, opt_g, opt_d)
                    g_acc += g
                    d_acc += d
                gl += g_acc / gb
                dl += d_acc / gb
            if (ep + 1) % 20 == 0 or ep == 0:
                print(f"[gan] epoch={ep+1} loss_g={gl/nsteps:.4f} loss_d={dl/nsteps:.4f}", flush=True)
            if val_ids and ((ep + 1) % cfg["val_every"] == 0 or ep + 1 == cfg["gan_epochs"]):
                ev = self._eval(model, videos, val_ids)
                mets = ev["metrics"]
                mae = float(mets.get("mae_all", float("nan")))
                mae2 = float(mets.get("mae_2", float("nan")))
                mae5 = float(mets.get("mae_5", float("nan")))
                pacc = float(ev.get("phase_acc", 0.0))
                score, mode, hi = self._select_score(mets, pacc, cfg["select_on"])
                better = (score > best) if hi else (score < best)
                if better:
                    best, bad = score, 0
                    torch.save(model.state_dict(), best_p)
                else:
                    bad += 1
                print(
                    f"[val] epoch={ep+1} mae_all={mae:.4f} mae_2={mae2:.4f} mae_5={mae5:.4f} "
                    f"phase_acc={pacc:.3f} best_{mode}={best:.4f} bad={bad}/{cfg['patience']}"
                    + (" **" if better else ""),
                    flush=True,
                )
                torch.save({"model": model.state_dict(), "epoch": ep + 1, "best": best, "bad": bad}, last_p)
                if cfg["patience"] > 0 and bad >= cfg["patience"]:
                    print(f"[early_stop] epoch={ep+1}", flush=True)
                    break
        if best_p.exists():
            model.load_state_dict(torch.load(best_p, map_location=self.device), strict=False)
        model.set_phase_refs(refs)
        return model

    @torch.no_grad()
    def _eval(self, model, videos, ids):
        model.eval()
        per, rows = {}, []
        correct, count = 0, 0
        for vid in ids:
            v = videos[vid]
            frames = read_frames(v["frames"])  # CPU
            elapsed = torch.from_numpy(v["elapsed"]).to(self.device)
            phase = torch.from_numpy(v["phase"]).to(self.device)
            frames_g = frames.to(self.device, non_blocking=True)
            out = model({"frames": frames_g, "elapsed": elapsed})
            pred = out["rsd"].detach().cpu().numpy().reshape(-1)
            if "phase_logits" in out:
                pred_ph = out["phase_logits"].argmax(-1)
                correct += int((pred_ph == phase).sum().item())
                count += int(phase.numel())
            del frames_g, out
            tgt = v["rsd"].reshape(-1)
            per[str(vid)] = {"pred": pred, "target": tgt}
            for i in range(len(pred)):
                rows.append({"video_id": str(vid), "t_idx": i, "elapsed_min": float(v["elapsed"][i]),
                             "rsd_true": float(tgt[i]), "rsd_pred": float(pred[i])})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return {
            "metrics": compute_dataset_metrics(per, windows=protocol_windows(self.configs)),
            "rows": rows,
            "phase_acc": correct / max(count, 1),
        }

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
            # refresh refs on full train
            model.set_phase_refs(phase_refs_from_videos(videos, list(rep["train"]), model.n_phases))
            result = self._eval(model, videos, rep["test"])
            ensure_dir(self.output_dir / rid)
            torch.save(model.state_dict(), self.output_dir / rid / "model.pth")
            save_predictions_csv(result["rows"], self.output_dir / rid / "predictions.csv")
            save_json(result["metrics"], self.output_dir / rid / "metrics.json")
            repeat_metrics[rid] = result["metrics"]
        return save_summary(self.output_dir, repeat_metrics)
