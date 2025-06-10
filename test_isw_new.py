import argparse
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from tqdm import tqdm

from dataset.dataset import SemanticKitti
from utils.utils import *
from model.model_isw import UNet, ISW_Loss
import pandas as pd
import os
import shutil
import time
import math
import random
import torch.nn.functional as F

parser = argparse.ArgumentParser("Train on semantic kitti")
parser.add_argument("--epoch", required=True, type=int)
parser.add_argument("--pts_path", type=Path)
parser.add_argument("--labs_path", type=Path)


args = parser.parse_args()

if os.path.exists("val_results_isw_test"):
    shutil.rmtree("val_results_isw_test")
os.makedirs("val_results_isw_test")


if os.path.exists("val_results_isw_test/img_res"):
    shutil.rmtree("val_results_isw_test/img_res")
os.makedirs("val_results_isw_test/img_res")


if os.path.exists("val_results_isw_test/pcd"):
    shutil.rmtree("val_results_isw_test/pcd")
os.makedirs("val_results_isw_test/pcd")


sup_colors = {
    0: [0, 0, 0],       # car - black
    1: [0, 0, 1],       # bicycle - blue
    2: [1, 0, 0],       # motorcycle - red
    3: [1, 0, 1],       # truck - magenta
    4: [0, 1, 1],       # other-vehicle - cyan
    5: [0.5, 0.5, 0],   # person - olive
    6: [1, 0.5, 0],     # bicyclist - orange
    7: [1, 1, 0],       # motorcyclist - yellow
    8: [1, 0, 0.5],     # road - pink
    9: [0.5, 0.5, 0.5], # parking - gray
    10: [0.5, 0, 0],    # sidewalk - dark red
    11: [0, 0.5, 0],    # other-ground - dark green
    12: [0, 0, 0.5],    # building - dark blue
    13: [0, 0.5, 0.5],  # fence - teal
    14: [0.5, 0, 0.5],  # vegetation - purple
    15: [0, 1, 0],      # trunk - green
    16: [0.7, 0.7, 0.7],# terrain - light gray
    17: [0.7, 0, 0.7],  # pole - light purple
    18: [0, 0.7, 0.7],  # traffic-sign - light cyan
    19: [0.7, 0.7, 0]   # unlabeled - light yellow
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


def batch_back_project(cl_map_2d, rows, cols):
    lab_3d = []
    for i in range(len(rows)):
        curr_row = rows[i].to(cl_map_2d[i].device)
        curr_col = cols[i].to(cl_map_2d[i].device)
        curr_cl_map_2d = cl_map_2d[i].squeeze(0)
        lab_3d_i = curr_cl_map_2d[curr_row, curr_col]
        lab_3d.append(lab_3d_i)
    return lab_3d


def custom_collate(batch):
    collated = {}
    for key in batch[0]:
        items = [d[key] for d in batch]
        try:
            # Try to stack (e.g., for depth_image, label_image)
            collated[key] = torch.stack(items)
        except:
            # If stacking fails (e.g., variable-length point clouds), keep as list
            collated[key] = items

    return collated


def eval_val(model, pts_pth, lab_pth, num_classes, epoch, fov_up_deg, fov_down_deg, H, W, save_vis=False):
    pts_files = os.listdir(pts_pth)
    labs_files = os.listdir(lab_pth)

    model.eval()
    with torch.no_grad():

        all_weighted_class_iou_records = []
        all_class_iou_records = []
        all_class_nums = []

        save_count = 0

        for idx, (pt_file, lab_file) in enumerate(zip(pts_files, labs_files)):
            points = np.load(os.path.join(pts_pth, pt_file))
            labs = np.load(os.path.join(lab_pth, lab_file))
            points_xyz = points[:, :3]
            points_refl = points[:, 3]

            (depth_image, refl_image, label_image, px, py , points_xyz, points_refl, labels) = spherical_projection(points_xyz,
                                                            points_refl,
                                                            labs,
                                                            fov_up_deg=fov_up_deg ,
                                                            fov_down_deg=fov_down_deg,
                                                            H= H ,
                                                            W= W
                                                            )
            
            items = {
                "depth_image": torch.from_numpy(depth_image).float().unsqueeze(0).unsqueeze(0),
                "reflectivity_image": torch.from_numpy(refl_image).float().unsqueeze(0).unsqueeze(0),
                "label_image": torch.from_numpy(label_image).long().unsqueeze(0),
                "px": [torch.from_numpy(px).long()],
                "py": [torch.from_numpy(py).long()],
                "points_xyz": torch.from_numpy(points_xyz).float().unsqueeze(0),
                "points_refl": torch.from_numpy(points_refl).float().unsqueeze(0),
                "labels": torch.from_numpy(labels).long().unsqueeze(0),
                }

            depth_image = items["depth_image"].cuda(non_blocking=True)
            reflectivity_image = items["reflectivity_image"].cuda(
                non_blocking=True)
            depth_image = depth_image
            reflectivity_image = reflectivity_image
            labels_2d = items["label_image"].cuda(non_blocking=True)
            py = items["py"]
            px = items["px"]
            points_xyz = items["points_xyz"]

            H_new =64
            W_new = 2048

            H_old = H
            W_old = W

            resized_depth_image = F.interpolate(depth_image, size =(H_new, W_new), mode = 'bicubic',align_corners=True)
            resized_reflectivity_image = F.interpolate(reflectivity_image, size =(H_new, W_new), mode = 'bicubic',align_corners=True)

            l_3d = batch_back_project(labels_2d, px, py)

            pre_predictions, _, _ = model(resized_depth_image, resized_reflectivity_image)
            predictions = F.interpolate(pre_predictions, size =(H_old, W_old), mode = 'bicubic',align_corners=True)


            prediction_2d = predictions.argmax(1)

            p_3d = batch_back_project(prediction_2d, px, py)

            for ab in range(len(p_3d)):
                p3d = p_3d[ab].cpu().numpy()
                l3d = l_3d[ab].cpu().numpy()

                class_nums = []
                uniq_classes, uniq_counts = np.unique(l3d, return_counts=True)
                for i in range(num_classes):
                    if i in uniq_classes:
                        class_nums.append(uniq_counts[uniq_classes == i][0])
                    else:
                        class_nums.append(0)

                all_class_nums.append(class_nums)

                if save_vis :
                    if random.random() < 1.1:
                        save_count += 1
                        pxyz = points_xyz[ab].cpu().numpy()
                        p_2d_img = prediction_2d[ab].cpu().numpy()
                        l_2d_img = labels_2d[ab].cpu().numpy()
                        depth_2d_img = depth_image[ab, 0, :, :].cpu().numpy()
                        r_2d_img = reflectivity_image[ab, 0, :, :].cpu().numpy()

                        p_2d_img = (p_2d_img - np.percentile(p_2d_img,2)) / \
                            (np.percentile(p_2d_img,100-2) - np.percentile(p_2d_img,2) + 1e-6)
                        l_2d_img = (l_2d_img - np.percentile(l_2d_img,2)) / \
                            (np.percentile(l_2d_img,100-2) - np.percentile(l_2d_img,2))
                        depth_2d_img = (depth_2d_img - np.percentile(depth_2d_img,2)) / \
                            (np.percentile(depth_2d_img,100-2) - np.percentile(depth_2d_img,2))
                        r_2d_img = (r_2d_img - np.percentile(r_2d_img,2)) / \
                            (np.percentile(r_2d_img,100-2) - np.percentile(r_2d_img,2))

                        plt.imsave(
                            f"val_results_isw_test/img_res/Sample_{save_count}_pred.png", p_2d_img, cmap='gray')
                        plt.imsave(
                            f"val_results_isw_test/img_res/Sample_{save_count}_gt.png", l_2d_img, cmap='gray')
                        plt.imsave(
                            f"val_results_isw_test/img_res/Sample_{save_count}_depth.png", depth_2d_img, cmap='gray')
                        plt.imsave(
                            f"val_results_isw_test/img_res/Sample_{save_count}_reflectivity.png", r_2d_img, cmap='gray')

                        pred_colors = np.zeros((len(p3d), 3), dtype=np.float32)
                        ref_colors  = np.zeros((len(l3d), 3), dtype=np.float32)

                        for label in range(num_classes):
                            pred_colors[p3d == label] = sup_colors[label]


                        for label in range(num_classes):
                            ref_colors[l3d == label] = sup_colors[label]

                        pcd_gt = o3d.geometry.PointCloud()
                        pcd_gt.points = o3d.utility.Vector3dVector(pxyz)
                        pcd_gt.colors = o3d.utility.Vector3dVector(ref_colors)

                        pcd_pred = o3d.geometry.PointCloud()
                        pcd_pred.points = o3d.utility.Vector3dVector(pxyz)
                        pcd_pred.colors = o3d.utility.Vector3dVector(pred_colors)

                        o3d.io.write_point_cloud(
                            f"val_results_isw_test/pcd/Sample_{save_count}_pred.pcd", pcd_pred)
                        o3d.io.write_point_cloud(
                            f"val_results_isw_test/pcd/Sample_{save_count}_gt.pcd", pcd_gt)
                        

                conf_matrix = np.zeros(
                    (num_classes, num_classes), dtype=np.uint64)
                classy_iou = [0] * num_classes
                normal_classy_iou = [0] * num_classes


                valid_preds = p3d
                valid_labels = l3d
                for t, p in zip(valid_labels, valid_preds):
                    if 0 <= t < num_classes and 0 <= p < num_classes:
                        conf_matrix[t, p] += 1

                class_pixel_counts = conf_matrix.sum(axis=1)
                total_pixel_count = class_pixel_counts.sum()

                weights = class_pixel_counts / total_pixel_count

        
                for i in range(num_classes):
                    TP = conf_matrix[i, i]
                    FP = conf_matrix[:, i].sum() - TP
                    FN = conf_matrix[i, :].sum() - TP
                    denom = TP + FP + FN
                    if denom == 0:
                        iou = float('nan')
                        classy_iou[i] = 0.0
                        normal_classy_iou[i] = 0.0

                    elif TP == 0:
                        iou = TP / denom
                        classy_iou[i] = 0.0
                        normal_classy_iou[i] = 0.0
                    else:
                        iou = TP / denom
                        classy_iou[i] = iou * weights[i]
                        normal_classy_iou[i] = iou


                # for i in range(len(classy_iou)):
                #     print(f"Class {class_names[i]} | IOU : {classy_iou[i]} | Fraction of Point Cloud : {weights[i]*100:.2f}%")

                all_weighted_class_iou_records.append(classy_iou)
                all_class_iou_records.append(normal_classy_iou)


    class_scores = np.array(all_class_iou_records)    # N x num_classes
    weighted_class_scores = np.array(all_weighted_class_iou_records)  # N x num_classes
    class_counts = np.array(all_class_nums)  # N x num_classes  

    total_num_points = np.sum(class_counts)
    with open(f"val_results_isw_test/vallog_epoch_{epoch}.txt", "w") as f:
        f.write("---------------------------------------------------------\n")
        f.write("Class Scores: \n")
        for i in range(num_classes):
            weights_class_i = class_counts[:, i]/(np.sum(class_counts[:,i]) + 1e-6)
            scores_class_i = class_scores[:, i]

            expected_score_class_i = np.sum(weights_class_i * scores_class_i)
            prob_class_i = np.sum(class_counts[:,i])/total_num_points  
            f.write(f".... Class {class_names[i]} | with mIoU: {expected_score_class_i:.4f} | Naturally Occurs with Probability : {prob_class_i:.4f}\n")

        mwiou = np.mean(np.sum(weighted_class_scores, axis=1))  
        f.write(f"Weighted mIOU overall: {mwiou:.4f}")
        f.write("---------------------------------------------------------\n")  




def train():
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = True
    pts_path = args.pts_path
    labs_path = args.labs_path
    num_classes = 20
    model = UNet(in_channels_coarse=1, in_channels_fine=1,
                 out_channels=num_classes)
    model.load_state_dict(torch.load(f"isw_checkpoints/isw_epoch{args.epoch}.pth"))
    torch.cuda.set_device(0)
    model.cuda()


    eval_val(model, pts_path, labs_path , num_classes, args.epoch,
            fov_up_deg=44.07,
            fov_down_deg=-45.73,
            H= 64 ,
            W= 1024,
            save_vis=True  
             )


def main() -> None:

    train()


if __name__ == "__main__":
    main()
