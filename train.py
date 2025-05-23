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

parser = argparse.ArgumentParser("Train on semantic kitti")
parser.add_argument("--semantic-kitti-dir", required=True, type=Path)

args = parser.parse_args()

if os.path.exists("checkpoints"):
    shutil.rmtree("checkpoints")
os.makedirs("checkpoints")

if os.path.exists("results"):
    shutil.rmtree("results")
os.makedirs("results")

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




def eval_val(model, val_loader, num_classes, epoch):

    model.eval()
    with torch.no_grad():
        dense_class_ious = []
        sparse_class_ious = []
        cols = []
        for i in range(num_classes):
            cols.append(f'IoU_{class_names[i]}')

        cols.append('mIoU')


        df = pd.DataFrame(columns=cols)
    

        for step, items in tqdm(enumerate(val_loader)):
            if step < 10:
                depth_image = items["depth_image"].cuda(non_blocking=True)
                reflectivity_image = items["reflectivity_image"].cuda(non_blocking=True)
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
                    



                    if ab ==0:
                        pxyz= points_xyz[ab].cpu().numpy()

                        pred_colors = np.zeros((len(p3d), 3), dtype=np.float32)
                        ref_colors = np.zeros((len(l3d), 3), dtype=np.float32)


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

                        o3d.io.write_point_cloud(f"results/Epoch_{epoch}_pred.pcd", pcd_pred)
                        o3d.io.write_point_cloud(f"results/Epoch_{epoch}_gt.pcd", pcd_gt)

                    # print(np.unique(p3d))
                    # print(np.unique(l3d))

                    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.uint64)
                    classy_iou = [0] * num_classes



                    valid_preds = p3d
                    valid_labels = l3d
                    for t, p in zip(valid_labels, valid_preds):
                        if 0 <= t < num_classes and 0 <= p < num_classes:
                            conf_matrix[t, p] += 1

                    class_pixel_counts = conf_matrix.sum(axis=1)
                    total_pixel_count = class_pixel_counts.sum()

                    weights = class_pixel_counts / total_pixel_count

                    # print("\n==== Per-Class IoU and mIoU ====")
                    weighted_ious = []
                    valid_count = 0
                    wrong_count = 0
                    for i in range(num_classes):
                        TP = conf_matrix[i, i]
                        FP = conf_matrix[:, i].sum() - TP
                        FN = conf_matrix[i, :].sum() - TP
                        denom = TP + FP + FN
                        if denom == 0:
                            iou = float('nan')
                            # print(f"{class_names[i]:<15}: IoU = N/A (class absent in prediction and ground-truth)")
                            classy_iou[i] = 0.0

                        elif TP  == 0:
                            iou = TP / denom
                            wrong_count += 1
                            # print(f"{class_names[i]:<15}: IoU = {iou:.4f} (This class is predicted but not in ground-truth)")
                            classy_iou[i] = 0.0
                        else:
                            iou = TP / denom
                            valid_count += 1
                            weighted_ious.append(iou* weights[i])
                            # print(f"{class_names[i]:<15}: IoU = {iou:.4f}")
                            classy_iou[i] = iou

                    mask_classy_iou = (np.array(classy_iou) != 0.0)

                    mean_iou = sum(weighted_ious)

                    if np.sum(mask_classy_iou)==1:
                        sparse_class_ious.append(mean_iou)
                    else:
                        dense_class_ious.append(mean_iou)

                    # print(f"\nMean IoU over {valid_count}/{num_classes} valid classes (classes both in predictions and ground-truth): {mean_iou:.4f}")
                    
                    classy_iou.append(mean_iou)

                    df = df.append(pd.Series(classy_iou, index=df.columns), ignore_index=True)

            else:
                break

            
        df_np = df.to_numpy()
        per_classes_ious = []
        for i in range(0, num_classes):
            per_class_iou = df_np[:, i]
            mask_iou = per_class_iou != 0
            per_class_iou = per_class_iou[mask_iou]
            mean_iou = np.mean(per_class_iou)
            per_classes_ious.append(mean_iou)
        per_classes_ious = np.array(per_classes_ious)
        dense_mean_iou = np.mean(dense_class_ious)
        sparse_mean_iou = np.mean(sparse_class_ious)
        print("-------------------------------------------")
        print("===========Mean Scores=====================")
        for i in range(num_classes):
            print(f"Class {class_names[i]}: {per_classes_ious[i]:.4f}" if str(per_classes_ious[i]) != 'nan' else f"Class {class_names[i]}: N/A")
        print("-------------------------------------------")
        print(f"Mean IoU over fully annotated: {dense_mean_iou:.4f}")
        print(f"Mean IoU over partially annotated: {sparse_mean_iou:.4f}")
        print("===========================================")
        print("-------------------------------------------")




def train():
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = True
    num_classes = 20

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
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.00001, momentum=0.9, weight_decay=1e-4
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

            predictions = model(depth_image, reflectivity_image)

            # print(predictions.shape)
            # print(labels_2d.shape)
            # print(torch.min(labels_2d),torch.max(labels_2d))


            loss = loss_fn(predictions, labels_2d)
            optimizer.zero_grad()
            loss.backward()
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
            val_l = loss_val(model, val_loader, loss_fn)

            print(
                f"Epoch: {epoch}  Loss: {run_loss} | Grad Norm: {run_grad_norm} | Val Loss: {val_l} | lr: {scheduler.get_last_lr()}"
            )

            eval_val(model, val_loader, num_classes, epoch)

            torch.save(
                model.state_dict(), f"checkpoints/epoch{epoch}.pth"
            )
            


def main() -> None:

    train()


if __name__ == "__main__":
    main()
