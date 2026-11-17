"""Rivoir 2019 training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from exp.exp_basic import ExpBasic
from exp.exp_rivoir_common import gpu_window, load_videos, read_frames
from models.Rivoir2019 import corridor_smooth_l1
from utils.io import ensure_dir, save_json, save_predictions_csv, save_summary
from utils.metrics import compute_dataset_metrics, protocol_windows
from utils.seed import set_seed
from utils.splits import build_protocol_splits, save_splits

def _init_uniform_labels(T: int, n_sub: int) -> np.ndarray:
    edges = np.linspace(0, T, n_sub + 1).astype(np.int64)
    lab = np.zeros(T, dtype=np.int64)
    for k in range(n_sub):
        lab[edges[k] : edges[k + 1]] = k
    if T:
        lab[-1] = n_sub - 1
    return lab

def _resegment_from_logits(logits: np.ndarray, n_sub: int) -> np.ndarray:
    """Force contiguous n_sub segments via DP on cumulative log-prob."""
    T, C = logits.shape
    n_sub = min(n_sub, C, max(T, 1))
    logp = logits - logits.max(axis=1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(axis=1, keepdims=True) + 1e-12)
    pref = np.cumsum(logp, axis=0)

    def seg_score(i, j, k):
        if j <= i:
            return -1e18
        s = pref[j - 1, k] - (pref[i - 1, k] if i > 0 else 0.0)
        return float(s)

    dp = np.full((n_sub + 1, T + 1), -1e18)
    bp = np.full((n_sub + 1, T + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0
    for k in range(1, n_sub + 1):
        for j in range(k, T + 1):
            best, arg = -1e18, -1
            i0 = max(k - 1, j - max(T // n_sub * 3, 8))
            for i in range(i0, j):
                sc = dp[k - 1, i] + seg_score(i, j, k - 1)
                if sc > best:
                    best, arg = sc, i
            dp[k, j] = best
            bp[k, j] = arg
    lab = np.zeros(T, dtype=np.int64)
    j = T
    for k in range(n_sub, 0, -1):
        i = int(bp[k, j])
        if i < 0:
            i = k - 1
        lab[i:j] = k - 1
        j = i
    return lab

def _tc_score(pred: np.ndarray, n_sub: int) -> float:
    T = len(pred)
    if T == 0:
        return 0.0
    lengths, order = [], []
    i = 0
    while i < T:
        j = i + 1
        while j < T and pred[j] == pred[i]:
            j += 1
        lengths.append(j - i)
        order.append(int(pred[i]))
        i = j
    ref = np.zeros(T, dtype=np.int64)
    pos = 0
    for L, oid in zip(lengths, order):
        ref[pos : pos + L] = oid
        pos += L
    return float((pred == ref).mean())

def _contrastive_l2(a: torch.Tensor, b: torch.Tensor, positive: bool, margin: float) -> torch.Tensor:
    """Contrastive L2; a,b: (B,D)."""
    dist2 = ((a - b) ** 2).sum(-1)
    if positive:
        return dist2.mean()
    return torch.clamp(margin - dist2, min=0.0).mean()

class ExpRivoir2019(ExpBasic):
    def __init__(self, configs, repo_root):
        super().__init__(configs)
        self.repo_root = repo_root
        self.splits = build_protocol_splits(configs, repo_root)
        ensure_dir(self.output_dir)
        save_splits(self.splits, self.output_dir / "splits.json")
        self.model = self._build_model()

    def _cfg(self):
        m = self.configs.get("model") or {}
        es = self.configs.get("early_stopping") or {}
        return {
            "lr": float(self.configs.get("lr", 1e-5)),
            "wd": float(self.configs.get("weight_decay", 1e-5)),
            "epochs": int(self.configs.get("epochs", 200)),
            "seg_iters": int(m.get("seg_iters", 8)),
            "seg_epochs": int(m.get("seg_epochs", 5)),
            "n_sub": int(m.get("n_subactivities", 10)),
            "train_window": int(m.get("train_window", 384)),
            "patience": int(es.get("patience", 20)),
            "use_corridor": bool(m.get("use_corridor", True)),
            "lambda_sub": float(m.get("lambda_sub", 0.5)),
            "batch_windows": int(m.get("batch_windows", 2)),
            "steps_per_epoch": int(m.get("steps_per_epoch", 200)),
            "end_loss_gamma": float(m.get("end_loss_gamma", 0.0)),
            "end_window_bias": float(m.get("end_window_bias", 0.0)),
            "select_on": str(m.get("select_on", "mae_all")),
            "pipeline": str(m.get("pipeline", "regularization")),
            "pretrained_lr_mult": float(m.get("pretrained_lr_mult", 0.1)),
            "tc_epochs": int(m.get("tc_epochs", 5)),
            "tc_steps": int(m.get("tc_steps", 200)),
            "tc_batch": int(m.get("tc_batch", 16)),
            "tc_gap": int(m.get("tc_gap", 1)),
            "tc_margin": float(m.get("tc_margin", 1.0)),
            "tc_lambda_steady": float(m.get("tc_lambda_steady", 1.0)),
        }

    def _window(self, T: int, win: int, end_bias: float = 0.0):
        if T <= win:
            return 0, T
        max_s = T - win
        if end_bias > 0 and np.random.rand() < float(end_bias):
            lo = max(0, max_s // 2)
            s = int(np.random.randint(lo, max_s + 1))
        else:
            s = int(np.random.randint(0, max_s + 1))
        return s, s + win

    def _sample_tc_batch(self, videos, train_ids, cfg):
        """Sample slow pairs + steady pos/neg triplets from 1fps videos."""
        gap = max(1, cfg["tc_gap"])
        B = max(1, cfg["tc_batch"])
        frames_p, frames_q = [], []
        trip_a, trip_b, trip_c, trip_an, trip_bn, trip_cn = [], [], [], [], [], []
        ids = list(train_ids)
        for _ in range(B):
            vid = ids[int(np.random.randint(0, len(ids)))]
            fr = read_frames(videos[vid]["frames"])
            T = fr.size(0)
            if T < gap * 4 + 3:
                t = max(0, T // 2)
                x = fr[t : t + 1]
                frames_p.append(x)
                frames_q.append(x)
                trip_a.append(x)
                trip_b.append(x)
                trip_c.append(x)
                trip_an.append(x)
                trip_bn.append(x)
                trip_cn.append(x)
                continue
            t0 = int(np.random.randint(0, T - gap))
            frames_p.append(fr[t0 : t0 + 1])
            frames_q.append(fr[t0 + gap : t0 + gap + 1])
            t1 = int(np.random.randint(0, T - 2 * gap))
            trip_a.append(fr[t1 : t1 + 1])
            trip_b.append(fr[t1 + gap : t1 + gap + 1])
            trip_c.append(fr[t1 + 2 * gap : t1 + 2 * gap + 1])
            t2 = int(np.random.randint(0, T - gap - 1))
            far = min(T - 1, t2 + gap + max(2 * gap, int(0.25 * T)))
            far = max(far, t2 + gap + 1)
            trip_an.append(fr[t2 : t2 + 1])
            trip_bn.append(fr[t2 + gap : t2 + gap + 1])
            trip_cn.append(fr[far : far + 1])

        def cat(xs):
            return torch.cat(xs, 0).to(self.device, non_blocking=True)

        return {
            "p": cat(frames_p),
            "q": cat(frames_q),
            "a": cat(trip_a),
            "b": cat(trip_b),
            "c": cat(trip_c),
            "an": cat(trip_an),
            "bn": cat(trip_bn),
            "cn": cat(trip_cn),
        }

    def _pretrain_temporal_coherence(self, model, videos, train_ids, ckpt: Path, cfg):
        """2nd-order temporal coherence (slow + steady) on CNN only."""
        done = ckpt / "tc_DONE"
        path = ckpt / "tc_cnn.pth"
        if done.exists() and path.exists() and bool(self.configs.get("resume", True)):
            model.cnn.load_state_dict(torch.load(path, map_location=self.device))
            print(f"[resume] temporal coherence → {path}", flush=True)
            return
        for p in model.parameters():
            p.requires_grad = False
        for p in model.cnn.parameters():
            p.requires_grad = True
        opt = torch.optim.Adam(model.cnn.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        margin = cfg["tc_margin"]
        lam = cfg["tc_lambda_steady"]
        model.train()
        for ep in range(cfg["tc_epochs"]):
            total, n = 0.0, 0
            for _ in tqdm(range(cfg["tc_steps"]), desc=f"tc {ep+1}/{cfg['tc_epochs']}", leave=False):
                batch = self._sample_tc_batch(videos, train_ids, cfg)
                zp = model.cnn(batch["p"])
                zq = model.cnn(batch["q"])
                loss_slow = _contrastive_l2(zp, zq, positive=True, margin=margin)
                za, zb, zc = model.cnn(batch["a"]), model.cnn(batch["b"]), model.cnn(batch["c"])
                zan, zbn, zcn = model.cnn(batch["an"]), model.cnn(batch["bn"]), model.cnn(batch["cn"])
                d_pos = za - zb
                d_pos2 = zb - zc
                d_neg = zan - zbn
                d_neg2 = zbn - zcn
                loss_steady = _contrastive_l2(d_pos, d_pos2, positive=True, margin=margin) + _contrastive_l2(
                    d_neg, d_neg2, positive=False, margin=margin
                )
                loss = loss_slow + lam * loss_steady
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.item())
                n += 1
                del batch, zp, zq, za, zb, zc, zan, zbn, zcn
            print(f"[tc] epoch={ep+1} loss={total/max(n,1):.4f}", flush=True)
        torch.save(model.cnn.state_dict(), path)
        done.write_text("ok")
        print(f"[tc] saved {path}", flush=True)

    def _train_seg_iter(self, model, videos, train_ids, labels, seg_iter, cfg):
        model.configure_seg_trainable(seg_iter)
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return
        opt = torch.optim.Adam(params, lr=cfg["lr"], weight_decay=1e-4)
        model.train()
        ids = list(train_ids)
        for _ in range(cfg["seg_epochs"]):
            np.random.shuffle(ids)
            for vid in ids:
                v = videos[vid]
                frames = read_frames(v["frames"])
                elapsed = torch.from_numpy(v["elapsed"]).to(self.device)
                lab = torch.from_numpy(labels[vid]).to(self.device)
                s, e = self._window(frames.size(0), cfg["train_window"])
                out = model({"frames": gpu_window(frames, s, e, self.device), "elapsed": elapsed[s:e]})
                loss = F.cross_entropy(out["sub_logits"], lab[s:e])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                del out

    def _run_unsupervised_seg(self, model, videos, train_ids, out_dir: Path, cfg):
        n_sub = cfg["n_sub"]
        labels = {vid: _init_uniform_labels(len(videos[vid]["phase"]), n_sub) for vid in train_ids}
        best_tc, best_labels, best_state = -1.0, None, None
        for it in range(cfg["seg_iters"]):
            self._train_seg_iter(model, videos, train_ids, labels, it, cfg)
            tcs = []
            for vid in train_ids:
                frames = read_frames(videos[vid]["frames"])
                elapsed = torch.from_numpy(videos[vid]["elapsed"]).to(self.device)
                T = frames.size(0)
                chunk = 256
                logit_parts = []
                model.last_state = None
                for i in range(0, T, chunk):
                    j = min(T, i + chunk)
                    out = model(
                        {
                            "frames": gpu_window(frames, i, j, self.device),
                            "elapsed": elapsed[i:j],
                            "stateful": True,
                        }
                    )
                    logit_parts.append(out["sub_logits"].detach().cpu().numpy())
                    del out
                model.last_state = None
                logits = np.concatenate(logit_parts, 0)
                labels[vid] = _resegment_from_logits(logits, n_sub)
                tcs.append(_tc_score(labels[vid], n_sub))
            tc = float(np.mean(tcs))
            n_train = sum(1 for p in model.parameters() if p.requires_grad)
            print(
                f"[Rivoir2019 seg] iter={it+1}/{cfg['seg_iters']} TC={tc:.4f} trainable_tensors={n_train}",
                flush=True,
            )
            if it >= max(0, cfg["seg_iters"] - 3) and tc >= best_tc:
                best_tc = tc
                best_labels = {k: v.copy() for k, v in labels.items()}
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
            labels = best_labels
        torch.save({"labels": best_labels, "tc": best_tc}, out_dir / "seg_labels.pt")
        return labels

    def _train_rsd(self, model, videos, train_ids, val_ids, labels, ckpt_dir: Path, cfg):
        pipe = cfg["pipeline"]
        model.configure_rsd_trainable(pipe)
        use_aux = pipe.lower().replace("-", "_") in ("regularization", "reg")
        groups = model.rsd_param_groups(cfg["lr"], cfg["pretrained_lr_mult"])
        if pipe.lower().replace("-", "_") in ("pretraining", "pt"):
            opt = torch.optim.SGD(groups, momentum=0.9, weight_decay=cfg["wd"])
        else:
            opt = torch.optim.Adam(groups, weight_decay=cfg["wd"])
        print(
            f"[rsd] pipeline={pipe} corridor={cfg['use_corridor']} aux_sub={use_aux} "
            f"param_groups={[(len(g['params']), g['lr']) for g in groups]}",
            flush=True,
        )

        t_median = float(np.median([float(videos[v]["elapsed"][-1]) for v in train_ids]))
        best, bad = float("inf"), 0
        best_p = ckpt_dir / "best.pt"
        last_p = ckpt_dir / "last.pt"
        start = 0
        if last_p.exists() and bool(self.configs.get("resume", True)):
            st = torch.load(last_p, map_location=self.device)
            model.load_state_dict(st["model"])
            try:
                opt.load_state_dict(st["optim"])
            except Exception:
                pass
            start, best, bad = int(st["epoch"]), float(st["best"]), int(st["bad"])
            print(f"[resume] RSD epoch={start} best={best:.4f}", flush=True)

        bw = max(1, cfg["batch_windows"])
        steps_per_ep = max(cfg["steps_per_epoch"], max(1, int(np.ceil(len(train_ids) / bw)) * 2))
        select_on = cfg["select_on"]
        for ep in range(start, cfg["epochs"]):
            model.train()
            total, n = 0.0, 0
            for _ in tqdm(range(steps_per_ep), desc=f"rsd {ep+1}/{cfg['epochs']}", leave=False):
                opt.zero_grad(set_to_none=True)
                loss_acc = 0.0
                for _j in range(bw):
                    vid = train_ids[int(np.random.randint(0, len(train_ids)))]
                    v = videos[vid]
                    frames = read_frames(v["frames"])
                    elapsed = torch.from_numpy(v["elapsed"]).to(self.device)
                    rsd = torch.from_numpy(v["rsd"]).to(self.device)
                    s, e = self._window(frames.size(0), cfg["train_window"], cfg["end_window_bias"])
                    tgt = rsd[s:e]
                    out = model({"frames": gpu_window(frames, s, e, self.device), "elapsed": elapsed[s:e]})
                    gamma = cfg["end_loss_gamma"]
                    w = (1.0 + gamma / (tgt.detach().abs() + 0.5)) if gamma > 0 else None
                    if cfg["use_corridor"]:
                        loss_r = corridor_smooth_l1(out["rsd"], tgt, elapsed[s:e], t_median, weights=w)
                    elif w is not None:
                        loss_r = (w * F.smooth_l1_loss(out["rsd"], tgt, reduction="none")).mean()
                    else:
                        loss_r = F.smooth_l1_loss(out["rsd"], tgt)
                    loss = loss_r
                    if use_aux:
                        lab = torch.from_numpy(labels[vid]).to(self.device)
                        loss = loss + cfg["lambda_sub"] * F.cross_entropy(out["sub_logits"], lab[s:e])
                    (loss / bw).backward()
                    loss_acc += float(loss.item())
                    del out
                opt.step()
                total += loss_acc / bw
                n += 1
            print(f"[train] epoch={ep+1} loss={total/max(n,1):.4f}", flush=True)
            vm = self._eval(model, videos, val_ids)["metrics"]
            mae = float(vm.get("mae_all", float("nan")))
            mae2 = float(vm.get("mae_2", float("nan")))
            mae5 = float(vm.get("mae_5", float("nan")))
            score = float(vm.get(select_on, mae))
            improved = score < best
            if improved:
                best, bad = score, 0
                torch.save(model.state_dict(), best_p)
            else:
                bad += 1
            print(
                f"[val] epoch={ep+1} mae_all={mae:.4f} mae_2={mae2:.4f} mae_5={mae5:.4f} "
                f"best_{select_on}={best:.4f} bad={bad}/{cfg['patience']}"
                + (" **" if improved else ""),
                flush=True,
            )
            torch.save(
                {"model": model.state_dict(), "optim": opt.state_dict(), "epoch": ep + 1, "best": best, "bad": bad},
                last_p,
            )
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
            frames = read_frames(v["frames"])
            elapsed = torch.from_numpy(v["elapsed"]).to(self.device)
            T = frames.size(0)
            chunk = 256
            preds = []
            model.last_state = None
            for i in range(0, T, chunk):
                j = min(T, i + chunk)
                out = model(
                    {
                        "frames": gpu_window(frames, i, j, self.device),
                        "elapsed": elapsed[i:j],
                        "stateful": True,
                    }
                )
                preds.append(out["rsd"].detach().cpu().numpy().reshape(-1))
                del out
            model.last_state = None
            pred = np.concatenate(preds, 0)
            tgt = v["rsd"].reshape(-1)
            per[str(vid)] = {"pred": pred, "target": tgt}
            for i in range(len(pred)):
                rows.append(
                    {
                        "video_id": str(vid),
                        "t_idx": i,
                        "elapsed_min": float(v["elapsed"][i]),
                        "rsd_true": float(tgt[i]),
                        "rsd_pred": float(pred[i]),
                    }
                )
        metrics = compute_dataset_metrics(per, windows=protocol_windows(self.configs))
        return {"metrics": metrics, "rows": rows}

    def _train_one(self, model, train_ids, val_ids=None):
        cfg = self._cfg()
        all_ids = list(dict.fromkeys(list(train_ids) + list(val_ids or [])))
        videos = load_videos(self.configs, all_ids)
        ckpt = Path(getattr(self, "_ckpt_dir"))
        ensure_dir(ckpt)
        self._pretrain_temporal_coherence(model, videos, list(train_ids), ckpt, cfg)
        seg_p = ckpt / "seg_labels.pt"
        if seg_p.exists() and bool(self.configs.get("resume", True)):
            pack = torch.load(seg_p, map_location="cpu")
            labels = pack["labels"]
            print(f"[resume] unsupervised seg labels TC={pack.get('tc')}", flush=True)
        else:
            labels = self._run_unsupervised_seg(model, videos, list(train_ids), ckpt, cfg)
        for vid in val_ids or []:
            labels.setdefault(vid, _init_uniform_labels(len(videos[vid]["phase"]), cfg["n_sub"]))
        return self._train_rsd(model, videos, list(train_ids), list(val_ids or []), labels, ckpt, cfg)

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
