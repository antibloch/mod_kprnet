import argparse
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from tqdm import tqdm

from dataset.dataset import SemanticKitti
from utils.utils import *
from model.model import UNet
import pandas as pd
import os
import shutil
import time
import math
import random

parser = argparse.ArgumentParser("Train on semantic kitti")
parser.add_argument("--semantic-kitti-dir", required=True, type=Path)

args = parser.parse_args()

if os.path.exists("checkpoints"):
    shutil.rmtree("checkpoints")
os.makedirs("checkpoints")

if os.path.exists("results"):
    shutil.rmtree("results")
os.makedirs("results")

if os.path.exists("results/img_res"):
    shutil.rmtree("results/img_res")
os.makedirs("results/img_res")

if os.path.exists("results/pcd"):
    shutil.rmtree("results/pcd")
os.makedirs("results/pcd")

# lab => lab -1
sup_colors = {
0: [0, 0, 0],       # unlabeled - black
1: [0, 0, 1],       # car - blue
2: [1, 0, 0],       # bicycle - red
3: [1, 0, 1],       # motorcycle - magenta
4: [0, 1, 1],       # truck - cyan
5: [0.5, 0.5, 0],   # other-vehicle - olive
6: [1, 0.5, 0],     # person - orange
7: [1, 1, 0],       # bicyclist - yellow
8: [1, 0, 0.5],     # motorcyclist - pink
9: [0.5, 0.5, 0.5], # road - gray
10: [0.5, 0, 0],    # parking - dark red
11: [0, 0.5, 0],    # sidewalk - dark green
12: [0, 0, 0.5],    # other-ground - dark blue
13: [0, 0.5, 0.5],  # building - teal
14: [0.5, 0, 0.5],  # fence - purple
15: [0, 1, 0],      # vegetation - green
16: [0.7, 0.7, 0.7],# trunk - light gray
17: [0.7, 0, 0.7],  # terrain - light purple
18: [0, 0.7, 0.7],  # pole - light cyan
19: [0.7, 0.7, 0]   # traffic-sign - light yellow
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


def loss_val(model, val_loader, loss_fn):
    model.eval()
    with torch.no_grad():
        average_loss = 0
        for step, items in tqdm(enumerate(val_loader)):
            if step < 10:
                depth_image = items["depth_image"].cuda(non_blocking=True)
                reflectivity_image = items["reflectivity_image"].cuda(non_blocking=True)
                depth_image = depth_image
                reflectivity_image = reflectivity_image
                labels_2d = items["label_image"].cuda(non_blocking=True)
                predictions = model(depth_image, reflectivity_image)

                loss = loss_fn(predictions, labels_2d)
                average_loss += loss.item()

            else:
                break

        average_loss /= step

    return average_loss




def eval_val(model, val_loader, num_classes, epoch, run_loss, run_grad_norm, val_l, lr_last, save_vis= False):

    model.eval()
    with torch.no_grad():


        all_weighted_class_iou_records = []
        all_class_iou_records = []
        all_class_nums = []

        save_count = 0

        for step, items in tqdm(enumerate(val_loader)):
            if step < 2:
                depth_image = items["depth_image"].cuda(non_blocking=True)
                reflectivity_image = items["reflectivity_image"].cuda(
                    non_blocking=True)
                depth_image = depth_image
                reflectivity_image = reflectivity_image
                labels_2d = items["label_image"].cuda(non_blocking=True)
                py = items["py"]
                px = items["px"]
                points_xyz = items["points_xyz"]
                l_3d = batch_back_project(labels_2d, px, py)

                predictions = model(depth_image, reflectivity_image)

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
                        if random.random() < 0.2:
                            save_count += 1
                            pxyz = points_xyz[ab].cpu().numpy()
                            p_2d_img = prediction_2d[ab].cpu().numpy()
                            l_2d_img = labels_2d[ab].cpu().numpy()
                            depth_2d_img = depth_image[ab, 0, :, :].cpu().numpy()
                            r_2d_img = reflectivity_image[ab, 0, :, :].cpu().numpy()

                            p_2d_img = (p_2d_img - np.min(p_2d_img)) / \
                                (np.max(p_2d_img) - np.min(p_2d_img) + 1e-6)
                            l_2d_img = (l_2d_img - np.min(l_2d_img)) / \
                                (np.max(l_2d_img) - np.min(l_2d_img) + 1e-6)
                            depth_2d_img = (depth_2d_img - np.min(depth_2d_img)) / \
                                (np.max(depth_2d_img) - np.min(depth_2d_img) + 1e-6)
                            r_2d_img = (r_2d_img - np.min(r_2d_img)) / \
                                (np.max(r_2d_img) - np.min(r_2d_img) + 1e-6)

                            plt.imsave(
                                f"results/img_res/Sample_{save_count}_pred.png", p_2d_img, cmap='gray')
                            plt.imsave(
                                f"results/img_res/Sample_{save_count}_gt.png", l_2d_img, cmap='gray')
                            plt.imsave(
                                f"results/img_res/Sample_{save_count}_depth.png", depth_2d_img, cmap='gray')
                            plt.imsave(
                                f"results/img_res/Sample_{save_count}_reflectivity.png", r_2d_img, cmap='gray')

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
                                f"results/pcd/Sample_{save_count}_pred.pcd", pcd_pred)
                            o3d.io.write_point_cloud(
                                f"results/pcd/Sample_{save_count}_gt.pcd", pcd_gt)
                            

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
                            classy_iou[i] = iou
                            normal_classy_iou[i] = iou * weights[i]


                    # for i in range(len(classy_iou)):
                    #     print(f"Class {class_names[i]} | IOU : {classy_iou[i]} | Fraction of Point Cloud : {weights[i]*100:.2f}%")

                    all_weighted_class_iou_records.append(classy_iou)
                    all_class_iou_records.append(normal_classy_iou)


    class_scores = np.array(all_class_iou_records)    # N x num_classes
    weighted_class_scores = np.array(all_weighted_class_iou_records)  # N x num_classes
    class_counts = np.array(all_class_nums)  # N x num_classes  

    total_num_points = np.sum(class_counts)
    with open(f"val_results/vallog_epoch_{epoch}.txt", "w") as f:
        f.write("---------------------------------------------------------\n")
        f.write("Class Scores: \n")
        for i in range(num_classes):
            weights_class_i = class_counts[:, i]/(np.sum(class_counts[:,i]) + 1e-6)
            scores_class_i = class_scores[:, i]

            expected_score_class_i = np.sum(weights_class_i * scores_class_i)
            prob_class_i = np.sum(class_counts[:,i])/total_num_points  
            f.write(f".... Class {class_names[i]} | with mIoU: {expected_score_class_i:.4f}\n | Naturally Occurs with Probability : {prob_class_i:.4f}\n")

        mwiou = np.mean(np.sum(weighted_class_scores, axis=1))  
        print(f"Weighted mIOU overall: {mwiou:.4f}")
        f.write(
            f"Epoch: {epoch}  Loss: {run_loss} | dL/d(theta) Norm: {run_grad_norm} | Val Loss: {val_l} | lr: {lr_last}\n"
        )

        f.write("---------------------------------------------------------\n")  




def check():

    train_dataset = SemanticKitti(
        args.semantic_kitti_dir / "dataset/sequences", "train",
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        collate_fn=custom_collate
    )


    depth_means= []
    depth_stds =[]

    refl_means = []
    refl_stds =[]
    for step, items in enumerate(train_loader):

        if step > 100:
            break

        depth_image = items["depth_image"].cuda(non_blocking=True)
        reflectivity_image = items["reflectivity_image"].cuda(non_blocking=True)
        depth_image = depth_image
        reflectivity_image = reflectivity_image

        dep_mean = depth_image.mean().item()
        dep_std = depth_image.std().item()


        ref_mean = reflectivity_image.mean().item()
        ref_std = reflectivity_image.std().item()

        depth_means.append(dep_mean)
        depth_stds.append(dep_std)

        refl_means.append(ref_mean)
        refl_stds.append(ref_std)

    print(f"Depth mean: {np.mean(depth_means)} | Depth std: {np.mean(depth_stds)}")
    print(f"Reflectivity mean: {np.mean(refl_means)} | Reflectivity std: {np.mean(refl_stds)}")





def main() -> None:

    check()


if __name__ == "__main__":
    main()
