"""CataNet training (Marafioti et al., MICCAI 2021)."""

from __future__ import annotations

import glob
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomVerticalFlip,
    Resize,
    ToPILImage,
    ToTensor,
)
from tqdm import tqdm

from data_provider.datasets.catanet_frames import DatasetCataract101, list_split_folders
from exp.exp_rsd import ExpRSD
from models.CataNet import CatRSDNet, Model
from utils.io import ensure_dir, save_json, save_predictions_csv, save_summary
from utils.metrics import compute_dataset_metrics, protocol_windows
from utils.seed import set_seed


def _link_split(ids, pool: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for child in list(dest.iterdir()):
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
    for vid in ids:
        src = pool / str(vid)
        if not src.exists():
            raise FileNotFoundError(
                f"missing CataNet frame pool video {src}. "
                "Run scripts/cataract101/prepare_catanet_frames.py first."
            )
        (dest / str(vid)).symlink_to(src.resolve(), target_is_directory=True)


class ExpCataNet(ExpRSD):

    def _stage_cfg(self) -> Dict[str, Any]:
        m = self.configs.get("model") or {}
        stages = self.configs.get("stages") or m.get("stages") or {}
        # Defaults =        rnn_epochs = list(stages.get("rnn_epochs", [50, 20, 80]))
        rp = stages.get("rnn_patience", [15, 10, 15])
        if isinstance(rp, (list, tuple)):
            rnn_patiences = [int(x) for x in rp]
        else:
            rnn_patiences = [int(rp)] * len(rnn_epochs)
        if len(rnn_patiences) < len(rnn_epochs):
            rnn_patiences = rnn_patiences + [rnn_patiences[-1]] * (len(rnn_epochs) - len(rnn_patiences))
        return {
            "cnn_epochs": int(stages.get("cnn_epochs", 20)),
            "cnn_lr": float(stages.get("cnn_lr", 1e-4)),
            "cnn_batch_size": int(stages.get("cnn_batch_size", 50)),
            "val_batch_size": int(stages.get("val_batch_size", 150)),
            "cnn_weighted_loss": bool(stages.get("cnn_weighted_loss", True)),
            "sub_epoch_validation": int(stages.get("sub_epoch_validation", 100)),
            "cnn_patience": int(stages.get("cnn_patience", 10)),  # 0 = no early stop
            "rnn_epochs": rnn_epochs,
            "rnn_lrs": list(stages.get("rnn_lrs", [0.001, 0.0001, 0.0005])),
            "rnn_sequence": list(stages.get("rnn_sequence", ["train_rnn", "train_all", "train_rnn"])),
            "window": int(stages.get("window", self.configs.get("train_window", 48))),
            "exp_weight": float(stages.get("exp_weight", 0.3)),
            "n_step_classes": int(m.get("n_phases", 11)),
            "input_size": int(stages.get("input_size", 224)),
            "feature_batch": int(stages.get("feature_batch", 500)),
            # per RNN stage; 0 = fixed epochs (best-val ckpt still selected)
            "rnn_patiences": rnn_patiences,
        }

    _KNOWN_DATASETS = ("cataract101", "cholec80")

    def _dataset(self) -> str:
        dataset = str(self.configs.get("dataset") or "")
        if dataset not in self._KNOWN_DATASETS:
            raise ValueError(f"CataNet-style frame pool supports {self._KNOWN_DATASETS}, got dataset={dataset!r}")
        return dataset

    def _resolve_pool(self) -> Path:
        """Frame pool path for the configured dataset (default .cache/catanet_{dataset}/pool).

        Refuses a configured pool whose path names a different dataset.
        """
        root = Path(self.repo_root)
        dataset = self._dataset()
        explicit = self.configs.get("catanet_frame_pool") or (self.configs.get("model") or {}).get("frame_pool")
        pool = Path(explicit) if explicit else root / ".cache" / f"catanet_{dataset}" / "pool"
        if not pool.is_absolute():
            pool = root / pool
        other = [d for d in self._KNOWN_DATASETS if d != dataset and d in str(pool).lower()]
        if other:
            raise ValueError(f"dataset={dataset!r} but frame pool path looks like {other[0]!r}: {pool}")
        return pool

    def _ensure_pool(self, video_ids) -> Path:
        """Pool with every requested video present and complete; missing ones are built."""
        from data_provider.data_factory import _data_root
        from data_provider.datasets.catanet_pool import ensure_pool

        pool = self._resolve_pool()
        stages = self.configs.get("stages") or (self.configs.get("model") or {}).get("stages") or {}
        ensure_pool(
            self._dataset(),
            _data_root(self.configs, self._dataset()),
            pool,
            video_ids,
            sample_fps=float((self.configs.get("model") or {}).get("fps", 2.5)),
            workers=int(stages.get("pool_workers", 4)),
        )
        return pool

    def _ds_kwargs(self) -> Dict[str, float]:
        m = self.configs.get("model") or {}
        return {"max_len": float(m.get("max_len", 20.0)), "fps": float(m.get("fps", 2.5))}

    def _pool_and_layout(self, seed: int, train_ids, val_ids) -> Path:
        root = Path(self.repo_root)
        pool = self._ensure_pool(list(train_ids) + list(val_ids or []))
        layout = root / ".cache" / f"catanet_{self._dataset()}" / "layout" / f"seed{seed}"
        _link_split(train_ids, pool, layout / "train")
        _link_split(val_ids or [], pool, layout / "val")
        return layout

    def _build_model(self):
        return super()._build_model()

    def _train_cnn(self, model: Model, basepath: Path, out_cnn: Path, st: Dict[str, Any]) -> Path:
        device = self.device
        n_step = st["n_step_classes"]
        input_size = [st["input_size"], st["input_size"]]
        img_transform = {
            "train": Compose(
                [
                    ToPILImage(),
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                    RandomResizedCrop(size=input_size[0], scale=(0.4, 1.0), ratio=(1.0, 1.0)),
                    ToTensor(),
                ]
            ),
            "val": Compose([ToPILImage(), Resize(input_size), ToTensor()]),
        }
        loaders = {}
        for phase in ("train", "val"):
            folders, labels = list_split_folders(basepath, phase)
            if not folders:
                if phase == "val":
                    continue
                raise RuntimeError(f"no folders under {basepath}/{phase}")
            ds = DatasetCataract101(folders, img_transform=img_transform[phase], label_files=labels, **self._ds_kwargs())
            bs = st["cnn_batch_size"] if phase == "train" else st["val_batch_size"]
            loaders[phase] = DataLoader(
                ds, batch_size=bs, shuffle=(phase == "train"), num_workers=4, pin_memory=True
            )

        cnn = model.net.cnn
        cnn = cnn.to(device)
        optim = torch.optim.Adam(cnn.parameters(), lr=st["cnn_lr"])

        if st["cnn_weighted_loss"]:
            label_sum = np.zeros(n_step)
            for fname_label in glob.glob(os.path.join(str(basepath), "train", "**", "*.csv"), recursive=True):
                labels = np.genfromtxt(fname_label, delimiter=",", skip_header=1)[:, 1]
                for l in range(n_step):
                    label_sum[l] += np.sum(labels == l)
            loss_weights = 1 / label_sum
            loss_weights[label_sum == 0] = 0.0
            loss_weights = torch.tensor(loss_weights / np.max(loss_weights)).float().to(device)
        else:
            loss_weights = None
        criterion = nn.CrossEntropyLoss(weight=loss_weights)
        expertise_criterion = nn.CrossEntropyLoss()
        # Cholec80 has no surgeon experience label (use_experience=false) → phase CE only
        use_exp = bool((self.configs.get("model") or {}).get("use_experience", True))

        ensure_dir(out_cnn)
        ckpt = out_cnn / "catRSDNet_CNN.pth"
        done = out_cnn / "DONE"
        if done.exists() and ckpt.exists():
            print(f"[resume] CNN stage done → {ckpt}", flush=True)
            model.net.cnn.load_state_dict(torch.load(ckpt, map_location=device)["model_dict"])
            return ckpt

        sub_every = int(st.get("sub_epoch_validation", 100))

        def _val_mean() -> float:
            cnn.eval()
            with torch.no_grad():
                val_sub = torch.zeros(0).to(device)
                for img_v, label_v in loaders["val"]:
                    img_v = img_v.to(device)
                    step_label = label_v[:, 0].long().to(device)
                    expertise = (label_v[:, 2] - 1).long().to(device)
                    prediction, expertise_pred = cnn(img_v)
                    loss_v = criterion(prediction, step_label)
                    if use_exp:
                        loss_v = loss_v + expertise_criterion(expertise_pred, expertise)
                    val_sub = torch.cat((val_sub, loss_v.detach().view(1, -1)))
            return float(val_sub.cpu().numpy().mean())

        best_loss = float("inf")
        cnn_patience = int(st.get("cnn_patience", 0))
        bad = 0
        start_epoch = 0
        # Incomplete CNN: reuse best weights (e.g. mid-ep1 save) and continue from next epoch.
        # Fresh runs have no ckpt → start_epoch=0. Normal path is always from scratch.
        if ckpt.exists():
            state = torch.load(ckpt, map_location=device)
            cnn.load_state_dict(state["model_dict"])
            start_epoch = int(state.get("epoch", 0))
            if "val" in loaders:
                best_loss = _val_mean()
                print(
                    f"[resume] CNN warm-start from {ckpt} (saved@epoch={start_epoch}, "
                    f"val_loss={best_loss:.4f}) → continue epoch {start_epoch + 1}..{st['cnn_epochs']} "
                    f"(sub_epoch_validation={sub_every})",
                    flush=True,
                )
            else:
                print(
                    f"[resume] CNN warm-start from {ckpt} (saved@epoch={start_epoch}) "
                    f"→ continue epoch {start_epoch + 1}..{st['cnn_epochs']}",
                    flush=True,
                )

        for epoch in range(start_epoch, st["cnn_epochs"]):
            cnn.train()
            all_loss_train = torch.zeros(0).to(device)
            improved = False
            for ii, (img, labels) in enumerate(tqdm(loaders["train"], desc=f"cnn ep{epoch+1}", leave=False)):
                img = img.to(device)
                step_label = labels[:, 0].long().to(device)
                expertise = labels[:, 2].long().to(device) - 1
                prediction, expertise_pred = cnn(img)
                loss = criterion(prediction, step_label)
                if use_exp:
                    loss = loss + expertise_criterion(expertise_pred, expertise)
                optim.zero_grad()
                loss.backward()
                optim.step()
                all_loss_train = torch.cat((all_loss_train, loss.detach().view(1, -1)))

                if "val" in loaders and sub_every > 0 and ii % sub_every == 0:
                    val_mean = _val_mean()
                    if val_mean < best_loss:
                        best_loss = val_mean
                        improved = True
                        torch.save({"epoch": epoch + 1, "model_dict": cnn.state_dict()}, ckpt)
                        print(f"  [cnn] val_loss={val_mean:.4f} ** saved")
                    cnn.train()

            # Per-epoch val when sub_epoch_validation<=0 (Cholec); with sub_every>0 the
            if "val" in loaders and sub_every <= 0:
                val_mean = _val_mean()
                mark = ""
                if val_mean < best_loss:
                    best_loss = val_mean
                    improved = True
                    torch.save({"epoch": epoch + 1, "model_dict": cnn.state_dict()}, ckpt)
                    mark = " ** saved"
                print(f"  [cnn] epoch_end val_loss={val_mean:.4f}{mark}", flush=True)
                cnn.train()

            print(
                f"[cnn] epoch={epoch+1}/{st['cnn_epochs']} "
                f"train_loss={float(all_loss_train.cpu().numpy().mean()):.4f}"
                + (f" best_val={best_loss:.4f} bad={bad}/{cnn_patience}" if cnn_patience > 0 else "")
            )
            if cnn_patience > 0 and "val" in loaders:
                if improved:
                    bad = 0
                else:
                    bad += 1
                if bad >= cnn_patience:
                    print(f"[early_stop] cnn epoch={epoch+1} best_val={best_loss:.4f}", flush=True)
                    break

        if not ckpt.exists():
            torch.save({"epoch": st["cnn_epochs"], "model_dict": cnn.state_dict()}, ckpt)
        done.write_text("ok\n", encoding="utf-8")
        # reload best into model.net.cnn
        state = torch.load(ckpt, map_location=device)
        model.net.cnn.load_state_dict(state["model_dict"])
        return ckpt

    def _cached_features(self, model: Model, cache_dir: Path, sequences, img_transform, st) -> Dict[str, np.ndarray]:
        """CNN features per video; float32 .npy cache so a crash does not redo extraction."""
        ensure_dir(cache_dir)
        features: Dict[str, np.ndarray] = {}
        model.eval()
        for label_path, input_path in tqdm(sequences, desc="precompute CNN feats", leave=False):
            f = cache_dir / (Path(input_path.rstrip(os.sep)).name + ".npy")
            if f.exists():
                arr = np.load(f)
                if arr.dtype == np.float32:
                    features[input_path] = arr
                    continue
            data = DatasetCataract101([input_path], [label_path], img_transform=img_transform, **self._ds_kwargs())
            dataloader = DataLoader(data, batch_size=st["feature_batch"], shuffle=False, num_workers=1, pin_memory=True)
            chunks = []
            with torch.no_grad():
                for X, _ in dataloader:
                    chunks.append(model.net.cnn(X.float().to(self.device)).cpu().numpy())
            arr = np.concatenate(chunks)
            np.save(f, arr)
            features[input_path] = arr
        return features

    def _train_rnn(self, model: Model, basepath: Path, out_dir: Path, cnn_ckpt: Path, st: Dict[str, Any]) -> Path:
        device = self.device
        n_step = st["n_step_classes"]
        model.net.cnn.load_state_dict(torch.load(cnn_ckpt, map_location=device)["model_dict"])
        model.set_cnn_as_feature_extractor()
        model = model.to(device)

        img_transform = Compose(
            [ToPILImage(), Resize(st["input_size"]), ToTensor()]
        )
        sequences_path = {key: {} for key in ("train", "val")}
        for phase in ("train", "val"):
            folders, labels = list_split_folders(basepath, phase)
            sequences_path[phase]["label"] = labels
            sequences_path[phase]["video"] = [d if d.endswith(os.sep) else d + os.sep for d in folders]

        if st.get("cnn_weighted_loss", True):
            label_sum = np.zeros(n_step)
            for fname_label in sequences_path["train"]["label"]:
                labels = np.genfromtxt(fname_label, delimiter=",", skip_header=1)[:, 1]
                for l in range(n_step):
                    label_sum[l] += np.sum(labels == l)
            loss_weights = 1 / label_sum
            loss_weights[label_sum == 0] = 0.0
            loss_weights = torch.tensor(loss_weights / np.max(loss_weights)).float().to(device)
        else:
            loss_weights = None

        step_criterion = nn.CrossEntropyLoss(weight=loss_weights)
        experience_criterion = nn.CrossEntropyLoss()
        rsd_criterion = nn.L1Loss()

        ensure_dir(out_dir)
        out_ckpt = out_dir / "catRSDNet.pth"
        last_p = out_dir / "last.pt"
        progress_p = out_dir / "progress.json"
        training_steps = list(st["rnn_sequence"])
        features: Dict[str, np.ndarray] = {}
        stage_done = 0
        if progress_p.exists():
            import json

            stage_done = int(json.loads(progress_p.read_text()).get("stage_done", 0))
        rnn_patiences = list(st.get("rnn_patiences") or [0] * len(training_steps))

        for step_count, training_step in enumerate(training_steps):
            print(f"[rnn] stage={training_step} ({step_count+1}/{len(training_steps)})")
            if step_count < stage_done:
                print(f"[resume] rnn stage {step_count+1} already done — skip", flush=True)
                continue
            if step_count > 0 and out_ckpt.exists():
                checkpoint = torch.load(out_ckpt, map_location=device)
                model.net.load_state_dict(checkpoint["model_dict"])

            best_loss_on_val = float("inf")
            stop_epoch = int(st["rnn_epochs"][step_count])
            patience = int(rnn_patiences[step_count]) if step_count < len(rnn_patiences) else 0
            lr = float(st["rnn_lrs"][step_count])
            optim = torch.optim.Adam(
                [
                    {"params": model.net.rnn.parameters()},
                    {"params": model.net.cnn.parameters(), "lr": lr / 20},
                ],
                lr=lr,
            )
            start_epoch, bad = 0, 0
            if last_p.exists():
                last = torch.load(last_p, map_location=device)
                if int(last.get("stage", -1)) == step_count:
                    model.net.load_state_dict(last["model_dict"])
                    optim.load_state_dict(last["optim"])
                    start_epoch = int(last["epoch"])
                    best_loss_on_val = float(last["best"])
                    bad = int(last.get("bad", 0))
                    print(f"[resume] rnn stage {step_count+1} from epoch {start_epoch}", flush=True)

            if training_step == "train_rnn":
                if len(features) == 0:
                    sequences = list(
                        zip(
                            sequences_path["train"]["label"] + sequences_path["val"]["label"],
                            sequences_path["train"]["video"] + sequences_path["val"]["video"],
                        )
                    )
                    features = self._cached_features(
                        model, out_dir / f"feats_s{step_count}", sequences, img_transform, st
                    )
                model.freeze_cnn(True)
                model.freeze_rnn(False)
            elif training_step == "train_all":
                model.freeze_cnn(False)
                model.freeze_rnn(False)
                features = {}
            else:
                raise RuntimeError(f"training step {training_step} not implemented")

            for epoch in range(start_epoch, stop_epoch):
                all_loss = {key: torch.zeros(0).to(device) for key in ("train", "val")}
                for phase in ("train", "val"):
                    sequences = list(zip(sequences_path[phase]["label"], sequences_path[phase]["video"]))
                    if phase == "train":
                        model.train()
                        random.shuffle(sequences)
                        model.net.cnn.eval()
                    else:
                        model.eval()

                    if not sequences:
                        continue

                    for label_path, input_path in sequences:
                        if training_step in ("train_rnn", "train_fc"):
                            label = torch.tensor(np.genfromtxt(label_path, delimiter=",", skip_header=1)[:, 1:])
                            dataloader = [
                                (
                                    torch.tensor(features[input_path]).unsqueeze(0),
                                    label[: len(features[input_path]), :],
                                )
                            ]
                            skip_features = True
                        else:
                            data = DatasetCataract101(
                                [input_path], [label_path], img_transform=img_transform, **self._ds_kwargs()
                            )
                            dataloader = DataLoader(
                                data,
                                batch_size=st["window"],
                                shuffle=False,
                                num_workers=1,
                                pin_memory=True,
                            )
                            skip_features = False

                        for i, (X, y) in enumerate(dataloader):
                            if len(y.shape) > 2:
                                y = y.squeeze()
                            y_experience = torch.add(y[:, 2], -1).long().to(device)
                            y_rsd = (y[:, 5] / 60.0 / 25.0).float().to(device)
                            y_step = y[:, 0].long().to(device)
                            X = X.float().to(device)
                            with torch.set_grad_enabled(phase == "train"):
                                stateful = i > 0
                                step_prediction, experience_prediction, rsd_prediction = model.net.forwardRNN(
                                    X, stateful=stateful, skip_features=skip_features
                                )
                                loss_step = step_criterion(step_prediction, y_step)
                                rsd_prediction = rsd_prediction.squeeze(1)
                                loss_rsd = rsd_criterion(rsd_prediction, y_rsd)
                                # Cholec: use_experience=false → exp_weight forced 0 (head dim may stay n_experience=2)
                                use_exp = bool((self.configs.get("model") or {}).get("use_experience", True))
                                exp_w = float(st["exp_weight"]) if use_exp else 0.0
                                loss = loss_step + loss_rsd
                                if exp_w > 0:
                                    loss = loss + exp_w * experience_criterion(
                                        experience_prediction, y_experience
                                    )
                                if phase == "train":
                                    optim.zero_grad()
                                    loss.backward()
                                    optim.step()
                                all_loss[phase] = torch.cat((all_loss[phase], loss.detach().view(1, -1)))

                    if all_loss[phase].numel() > 0:
                        all_loss[phase] = all_loss[phase].cpu().numpy().mean()
                    else:
                        all_loss[phase] = 0.0

                val_l = float(all_loss["val"])
                print(
                    f"[rnn/{training_step}] epoch={epoch+1}/{stop_epoch} "
                    f"train={float(all_loss['train']):.4f} val={val_l:.4f}",
                    end="",
                )
                if val_l < best_loss_on_val:
                    best_loss_on_val = val_l
                    bad = 0
                    print("  **")
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_dict": model.net.state_dict(),
                            "remaining_steps": training_steps[step_count:],
                        },
                        out_ckpt,
                    )
                else:
                    bad += 1
                    print("")
                torch.save(
                    {
                        "stage": step_count,
                        "epoch": epoch + 1,
                        "model_dict": model.net.state_dict(),
                        "optim": optim.state_dict(),
                        "best": best_loss_on_val,
                        "bad": bad,
                    },
                    last_p,
                )
                if patience > 0 and bad >= patience:
                    print(f"[early_stop] rnn stage {step_count+1} epoch={epoch+1}", flush=True)
                    break

            import json

            progress_p.write_text(json.dumps({"stage_done": step_count + 1}), encoding="utf-8")

        if out_ckpt.exists():
            model.net.load_state_dict(torch.load(out_ckpt, map_location=device)["model_dict"])
        return out_ckpt

    def _train_one(self, model, train_ids, val_ids=None):
        st = self._stage_cfg()
        # recover seed from current split context if present
        seed = int(self.configs.get("_current_seed", 42))
        layout = self._pool_and_layout(seed, train_ids, val_ids)
        print(f"[CataNet] layout={layout} (four-stage schedule; LRs={st['rnn_lrs']})")
        rid_dir = self.output_dir / f"seed{seed}"
        ensure_dir(rid_dir)
        cnn_ckpt = self._train_cnn(model, layout, rid_dir / "cnn", st)
        self._train_rnn(model, layout, rid_dir / "rnn", cnn_ckpt, st)
        return model

    @torch.no_grad()
    def _eval(self, model, video_ids):
        model.eval()
        model.set_cnn_as_feature_extractor()
        st = self._stage_cfg()
        pool = self._ensure_pool(video_ids)
        img_transform = Compose([ToPILImage(), Resize(st["input_size"]), ToTensor()])
        per_video = {}
        rows = []
        for vid in video_ids:
            vdir = pool / str(vid)
            label = vdir / f"{vid}.csv"
            if not vdir.exists() or not label.exists():
                raise FileNotFoundError(vdir)
            ds = DatasetCataract101([str(vdir)], [str(label)], img_transform=img_transform, **self._ds_kwargs())
            loader = DataLoader(ds, batch_size=st["window"], shuffle=False, num_workers=1, pin_memory=True)
            preds = []
            tgts = []
            elapsed_list = []
            t_idx = 0
            for i, (X, y) in enumerate(loader):
                if len(y.shape) > 2:
                    y = y.squeeze()
                y_rsd = (y[:, 5] / 60.0 / 25.0).float()
                X = X.float().to(self.device)
                stateful = i > 0
                _, _, rsd_prediction = model.net.forwardRNN(X, stateful=stateful, skip_features=False)
                rsd_prediction = rsd_prediction.squeeze(1).detach().cpu()
                for j in range(len(rsd_prediction)):
                    preds.append(float(rsd_prediction[j]))
                    tgts.append(float(y_rsd[j]))
                    # elapsed approx from sample index @ 2.5fps
                    elapsed_list.append(t_idx / (2.5 * 60.0))
                    rows.append(
                        {
                            "video_id": str(vid),
                            "t_idx": t_idx,
                            "elapsed_min": elapsed_list[-1],
                            "rsd_true": float(y_rsd[j]),
                            "rsd_pred": float(rsd_prediction[j]),
                        }
                    )
                    t_idx += 1
            per_video[str(vid)] = {"pred": np.asarray(preds), "target": np.asarray(tgts)}
        metrics = compute_dataset_metrics(per_video, windows=protocol_windows(self.configs))
        return {"metrics": metrics, "rows": rows}

    def run(self) -> Dict[str, Any]:
        scheme = self.splits.get("scheme")
        repeat_metrics = {}
        if scheme not in ("repeated_holdout", "stratified_repeated_holdout"):
            raise ValueError(f"CataNet runner expects stratified_repeated_holdout, got {scheme}")
        for rep in self.splits["repeats"]:
            rid = f"seed{rep['seed']}"
            done = self.output_dir / rid / "metrics.json"
            if done.exists():
                import json

                print(f"[skip] {rid} already has metrics.json", flush=True)
                repeat_metrics[rid] = json.loads(done.read_text())
                continue
            set_seed(int(rep["seed"]))
            self.configs["_current_seed"] = int(rep["seed"])
            model = self._build_model()
            model = self._train_one(model, rep["train"], rep.get("val"))
            result = self._eval(model, rep["test"])
            ensure_dir(self.output_dir / rid)
            torch.save(model.net.state_dict(), self.output_dir / rid / "model.pth")
            save_predictions_csv(result["rows"], self.output_dir / rid / "predictions.csv")
            save_json(result["metrics"], self.output_dir / rid / "metrics.json")
            repeat_metrics[rid] = result["metrics"]
            for d in (self.output_dir / rid / "rnn").glob("feats_s*"):
                shutil.rmtree(d, ignore_errors=True)
        return save_summary(self.output_dir, repeat_metrics)
