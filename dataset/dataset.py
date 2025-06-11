from pathlib import Path
import cv2
import numpy as np
import torch
from scipy.spatial.ckdtree import cKDTree as kdtree
from utils.utils import spherical_projection
splits = {
    "train": [1, 2, 0, 3, 4, 5, 6, 7, 9, 10],
    "val": [8],
    "test": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
}
class SemanticKitti(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: Path, split: str,) -> None:
        self.split = split
        self.seqs = splits[split]
        self.dataset_dir = dataset_dir
        self.sweeps = []
        for seq in self.seqs:
            seq_str = f"{seq:0>2}"
            seq_path = dataset_dir / seq_str / "velodyne"
            for sweep in seq_path.iterdir():
                self.sweeps.append((seq_str, sweep.stem))
    def __getitem__(self, index):
        seq, sweep = self.sweeps[index]
        sweep_file = self.dataset_dir / seq / "velodyne" / f"{sweep}.bin"
        points = np.fromfile(sweep_file.as_posix(), dtype=np.float32)
        points = points.reshape((-1, 4))
        points_xyz = points[:, :3]
        points_refl = points[:, 3]
        if self.split != "test":
            labels_file = self.dataset_dir / seq / "labels" / f"{sweep}.label"
            labels = np.fromfile(labels_file.as_posix(), dtype=np.int32)
            lab_data = labels.reshape((-1))
            semantic_labels = lab_data  & 0xFFFF  # bitwise AND with 0xFFFF
            remap_dict = learning_map
            u_s_l = np.unique(semantic_labels)
            label = np.zeros((semantic_labels.shape[0]), dtype=np.int32)
            for i in range(len(u_s_l)):
                label[semantic_labels == u_s_l[i]] = remap_dict[u_s_l[i]]
            label[label == 255] = 19
            labels = label
        else:
            labels = np.zeros((points.shape[0]), dtype=np.int32)
        (depth_image, refl_image, label_image, px, py , points_xyz, points_refl, labels) = spherical_projection(points_xyz,
                                                                    points_refl,
                                                                    labels,
                                                                    fov_up_deg=2.0 ,
                                                                    fov_down_deg=-24.9,
                                                                    H= 64 ,
                                                                    W= 2048
                                                                    )
        # percentile clipping
        depth_image = np.clip(depth_image, np.percentile(depth_image, 2), np.percentile(depth_image, 98-2))
        refl_image = np.clip(refl_image, np.percentile(refl_image, 2), np.percentile(refl_image, 2))

        # min-max normalization
        depth_image = (depth_image - np.min(depth_image))/(np.max(depth_image) - np.min(depth_image) + 1e-6)
        refl_image = (refl_image - np.min(refl_image))/(np.max(refl_image) - np.min(refl_image) + 1e-6)

        # dist normalization
        d_mean = np.load('depth_mean.npy')
        d_std = np.load('depth_std.npy')

        r_mean = np.load('reflectivity_mean.npy')
        r_std = np.load('reflectivity_std.npy')

        depth_image = (depth_image - d_mean) /(d_std + 1e-6)
        refl_image = (refl_image - r_mean) / (r_std + 1e-6)
        
        # print(depth_image.shape, refl_image.shape, label_image.shape, px.shape, py.shape, points_xyz.shape, points_refl.shape, labels.shape)
        res = {
            "depth_image": torch.from_numpy(depth_image).float().unsqueeze(0),
            "reflectivity_image": torch.from_numpy(refl_image).float().unsqueeze(0),
            "label_image": torch.from_numpy(label_image).long(),
            "px": torch.from_numpy(px).long(),
            "py": torch.from_numpy(py).long(),
            "points_xyz": torch.from_numpy(points_xyz).float(),
            "points_refl": torch.from_numpy(points_refl).float(),
            "labels": torch.from_numpy(labels).long(),
        }
        if self.split in ["test", "val"]:
            res["seq"] = seq
            res["sweep"] = sweep
        return res
    def __len__(self):
        return len(self.sweeps)
learning_map = {
    0: 255,  # "unlabeled"
    1: 255,  # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 0,  # "car"
    11: 1,  # "bicycle"
    13: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 2,  # "motorcycle"
    16: 4,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 3,  # "truck"
    20: 4,  # "other-vehicle"
    30: 5,  # "person"
    31: 6,  # "bicyclist"
    32: 7,  # "motorcyclist"
    40: 8,  # "road"
    44: 9,  # "parking"
    48: 10,  # "sidewalk"
    49: 11,  # "other-ground"
    50: 12,  # "building"
    51: 13,  # "fence"
    52: 255,  # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 8,  # "lane-marking" to "road" ---------------------------------mapped
    70: 14,  # "vegetation"
    71: 15,  # "trunk"
    72: 16,  # "terrain"
    80: 17,  # "pole"
    81: 18,  # "traffic-sign"
    99: 255,  # "other-object" to "unlabeled" ----------------------------mapped
    252: 0,  # "moving-car" to "car" ------------------------------------mapped
    253: 6,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 5,  # "moving-person" to "person" ------------------------------mapped
    255: 7,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 4,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 4,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 3,  # "moving-truck" to "truck" --------------------------------mapped
    259: 4,  # "moving-other"-vehicle to "other-vehicle" ----------------mappe
}
class_names = [
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
    "unlabeled"
]
map_inv = {
    0: 10,  # "car"
    1: 11,  # "bicycle"
    2: 15,  # "motorcycle"
    3: 18,  # "truck"
    4: 20,  # "other-vehicle"
    5: 30,  # "person"
    6: 31,  # "bicyclist"
    7: 32,  # "motorcyclist"
    8: 40,  # "road"
    9: 44,  # "parking"
    10: 48,  # "sidewalk"
    11: 49,  # "other-ground"
    12: 50,  # "building"
    13: 51,  # "fence"
    14: 70,  # "vegetation"
    15: 71,  # "trunk"
    16: 72,  # "terrain"
    17: 80,  # "pole"
    18: 81,  # "traffic-sign
    255: 0,
}
