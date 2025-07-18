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

parser = argparse.ArgumentParser("Train on semantic kitti")
parser.add_argument("--semantic-kitti-dir", required=True, type=Path)

args = parser.parse_args()

if os.path.exists("isw_checkpoints"):
    shutil.rmtree("isw_checkpoints")
os.makedirs("isw_checkpoints")

if os.path.exists("isw_results"):
    shutil.rmtree("isw_results")
os.makedirs("isw_results")

if os.path.exists("isw_results/img_res"):
    shutil.rmtree("isw_results/img_res")
os.makedirs("isw_results/img_res")

if os.path.exists("isw_results/pcd"):
    shutil.rmtree("isw_results/pcd")
os.makedirs("isw_results/pcd")

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


def loss_val(model, val_loader, loss_fn, isw_loss_fn):
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
                predictions,  coarse_feats, fine_feats = model(depth_image, reflectivity_image)

                loss = loss_fn(predictions, labels_2d)

                coarse_isw_loss = torch.mean(torch.stack([isw_loss_fn(feat) for feat in coarse_feats]))
                fine_isw_loss = torch.mean(torch.stack([isw_loss_fn(feat) for feat in fine_feats]))
                isw_loss = 0.5 * coarse_isw_loss + 0.5 * fine_isw_loss

                net_loss = loss + 1e-3 * isw_loss
                average_loss += net_loss.item()

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

                predictions, _, _ = model(depth_image, reflectivity_image)

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
                                f"isw_results/img_res/Sample_{save_count}_pred.png", p_2d_img, cmap='gray')
                            plt.imsave(
                                f"isw_results/img_res/Sample_{save_count}_gt.png", l_2d_img, cmap='gray')
                            plt.imsave(
                                f"isw_results/img_res/Sample_{save_count}_depth.png", depth_2d_img, cmap='gray')
                            plt.imsave(
                                f"isw_results/img_res/Sample_{save_count}_reflectivity.png", r_2d_img, cmap='gray')

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
                                f"isw_results/pcd/Sample_{save_count}_pred.pcd", pcd_pred)
                            o3d.io.write_point_cloud(
                                f"isw_results/pcd/Sample_{save_count}_gt.pcd", pcd_gt)
                            

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
    with open(f"isw_results/vallog_epoch_{epoch}.txt", "w") as f:
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
        f.write(
            f"Epoch: {epoch}  Loss: {run_loss} | dL/d(theta) Norm: {run_grad_norm} | Val Loss: {val_l} | lr: {lr_last}\n"
        )

        f.write("---------------------------------------------------------\n")  





def train():
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = True
    num_classes = 20
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    model =  UNet(in_channels_coarse=1,in_channels_fine=1, out_channels=num_classes)
    torch.cuda.set_device(0)
    model.cuda()


    train_dataset = SemanticKitti(
        args.semantic_kitti_dir / "dataset/sequences", "train",
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        collate_fn=custom_collate
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=SemanticKitti(
            args.semantic_kitti_dir / "dataset/sequences", "val",
        ),
        batch_size=4,
        shuffle=False,
        collate_fn=custom_collate
    )

    loss_fn = nn.CrossEntropyLoss()
    isw_loss_fn = ISW_Loss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=5e-5, momentum=0.5, weight_decay=5e-4
    )
    warmup = 10
    epochs = 120
    # Use lambda scheduler with warmup and cosine annealing as in reference
    lambda0 = lambda cur_iter: (cur_iter + 1) / warmup if cur_iter < warmup else (
        0.5 * (1.0 + np.cos(np.pi * ((cur_iter - warmup) / (epochs - warmup))))
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    
    n_iter = 0
    for epoch in range(epochs):
        n_iter +=1
        model.train()
        run_loss = 0.0
        run_grad_norm = 0.0
        for step, items in enumerate(train_loader):
            depth_image = items["depth_image"].cuda(non_blocking=True)
            reflectivity_image = items["reflectivity_image"].cuda(non_blocking=True)
            depth_image = depth_image
            reflectivity_image = reflectivity_image
            labels_2d = items["label_image"].cuda(non_blocking=True)

            predictions, coarse_feats, fine_feats = model(depth_image, reflectivity_image)

            loss = loss_fn(predictions, labels_2d)

            coarse_isw_loss = torch.mean(torch.stack([isw_loss_fn(feat) for feat in coarse_feats]))
            fine_isw_loss = torch.mean(torch.stack([isw_loss_fn(feat) for feat in fine_feats]))
            isw_loss = 0.5 * coarse_isw_loss + 0.5 * fine_isw_loss

            net_loss = loss + 1e-3 * isw_loss
            
            optimizer.zero_grad()
            net_loss.backward()
            grad_norm =torch.sum(torch.tensor([torch.norm(p.grad) for p in model.parameters() if p.grad is not None])).item()
            run_grad_norm += grad_norm
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()

            scheduler.step()
            torch.cuda.empty_cache()

            run_loss += loss.item()
        run_loss /= step
        run_grad_norm /= step
        if epoch % 1 == 0:
            val_l = loss_val(model, val_loader, loss_fn, isw_loss_fn)

            lr_last = scheduler.get_last_lr()
            
            eval_val(model, val_loader, num_classes, epoch, run_loss, run_grad_norm, val_l, lr_last)

            torch.save(
                model.state_dict(), f"isw_checkpoints/isw_epoch{epoch}.pth"
            )
            


def main() -> None:

    train()


if __name__ == "__main__":
    main()
