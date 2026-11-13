"""Frame dataset for CataNet."""

from __future__ import annotations

import glob
import os
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch


class DatasetNoLabel(torch.utils.data.Dataset):
    def __init__(self, datafolders, img_transform=None, max_len=20, fps=2.5):
        super().__init__()
        self.datafolders = datafolders
        self.img_transform = img_transform
        self.max_len = max_len * fps * 60.0
        self.frame2min = 1 / (fps * 60.0)
        self.surgery_length = {}
        img_files: List[str] = []
        for d in datafolders:
            files = sorted(glob.glob(os.path.join(d, "*.png")))
            img_files += files
            patientID, frame = self._name2id(files[-1])
            self.surgery_length[patientID] = float(frame) * self.frame2min
        img_files = sorted(img_files)
        assert len(img_files) > 0, f"no png images found in {datafolders}"
        self.img_files = img_files
        self.nitems = len(self.img_files)

    def __getitem__(self, index):
        ret = cv2.imread(self.img_files[index])
        if ret is None:
            print("weird: ", self.img_files[index])
        img = cv2.cvtColor(cv2.imread(self.img_files[index]), cv2.COLOR_BGR2RGB)
        patientID, frame_number = self._name2id(self.img_files[index])
        elapsed_time = frame_number * self.frame2min
        rsd = self.surgery_length[patientID] - elapsed_time
        if self.img_transform is not None:
            img = self.img_transform(img)
        time_stamp = torch.ones((1, img.shape[1], img.shape[2])) * frame_number / self.max_len
        img = torch.cat((img, time_stamp.float()), dim=0)
        return img, elapsed_time, frame_number, rsd

    def __len__(self):
        return self.nitems

    def _name2id(self, filename):
        *patientID, frame = os.path.splitext(os.path.basename(filename))[0].split("_")
        patientID = "_".join(patientID)
        return patientID, int(frame) - 1


class DatasetCataract101(DatasetNoLabel):
    def __init__(self, datafolders, label_files, img_transform=None, max_len=20, fps=2.5):
        super().__init__(datafolders, img_transform, max_len, fps)
        assert len(label_files) == len(datafolders), "not the same number of data and label files"
        self.label_files = {}
        for f in label_files:
            patientID = os.path.splitext(os.path.basename(f))[0]
            self.label_files[patientID] = np.genfromtxt(f, delimiter=",", skip_header=1)[:, 1:]

    def __getitem__(self, index):
        img, elapsed_time, frame_number, rsd = super().__getitem__(index)
        patientID, frame_number = self._name2id(self.img_files[index])
        label = self.label_files[patientID][frame_number]
        return img, label


def list_split_folders(base_path: str | os.PathLike, phase: str) -> tuple[list[str], list[str]]:
    """Return (video_dirs, label_csvs) sorted."""
    base = os.path.join(str(base_path), phase)
    data_folders = sorted(glob.glob(os.path.join(base, "*")))
    # only directories (symlink targets of video ids)
    data_folders = [d for d in data_folders if os.path.isdir(d)]
    labels = sorted(glob.glob(os.path.join(base, "**", "*.csv"), recursive=True))
    return data_folders, labels
