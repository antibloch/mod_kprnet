import argparse
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from tqdm import tqdm

from dataset.dataset import SemanticKitti
from utils.utils import *
from model.model import UNet


parser = argparse.ArgumentParser("Train on semantic kitti")
parser.add_argument("--semantic-kitti-dir", required=True, type=Path)
parser.add_argument("--model-dir", required=True, type=Path)
parser.add_argument("--checkpoint-dir", required=True, type=Path)
args = parser.parse_args()


def run_val(model, val_loader, n_iter, writer):
    print("Runnign validation")
    model.eval()

    loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    eval_metric = utils.evaluation.Eval(19, 255)
    with torch.no_grad():
        average_loss = 0
        for step, items in tqdm(enumerate(val_loader)):
            images = items["image"].cuda(0, non_blocking=True)
            labels = items["labels"].long().cuda(0, non_blocking=True)
            py = items["py"].float().cuda(0, non_blocking=True)
            px = items["px"].float().cuda(0, non_blocking=True)
            pxyz = items["points_xyz"].float().cuda(0, non_blocking=True)
            knns = items["knns"].long().cuda(0, non_blocking=True)
            predictions = model(images, px, py, pxyz, knns)

            loss = loss_fn(predictions, labels)
            average_loss += loss.item()
            _, predictions_argmax = torch.max(predictions, 1)
            eval_metric.update(predictions_argmax.cpu().numpy(), labels.cpu().numpy())

        average_loss /= step
        miou, ious = eval_metric.getIoU()
        print(f"Iteration {n_iter} Average Val Loss: {average_loss}, mIou {miou}")
        print(f"Per class Ious {ious}")
        writer.add_scalar("val/val", average_loss, n_iter)
        writer.add_scalar("val/mIoU", miou, n_iter)


def train():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    num_classes = 19

    model =  UNet(in_channels_coarse=1,in_channels_fine=1, out_channels=num_classes)
    torch.cuda.set_device(0)
    model.cuda()


    train_dataset = SemanticKitti(
        args.semantic_kitti_dir / "dataset/sequences", "train",
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        num_workers=1,
        drop_last=True,
        shuffle=False,
        pin_memory=True,
        # sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=SemanticKitti(
            args.semantic_kitti_dir / "dataset/sequences", "val",
        ),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
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
        model.train()
        for step, items in enumerate(train_loader):
            depth_image = items["depth_image"].cuda(non_blocking=True)
            reflectivity_image = items["reflectivity_image"].cuda(non_blocking=True)
            print(depth_image.shape)
            print(reflectivity_image.shape)
            skdfn
            labels_2d = items["label_image"].cuda(non_blocking=True)
            py = items["py"].cuda(non_blocking=True)
            px = items["px"].cuda(non_blocking=True)
            pxyz = items["points_xyz"].cuda(non_blocking=True)
            points_refl = items["points_refl"].cuda(non_blocking=True)
            labels_3d = items["labels"].cuda(non_blocking=True)
            
            predictions = model(depth_image, reflectivity_image)

            loss = loss_fn(predictions, labels_2d)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()

            print(
                f"Epoch: {epoch} Iteration: {step} / {len(train_loader)} Loss: {loss.item()}"
            )

            scheduler.step()
            torch.cuda.empty_cache()

 
        if (epoch + 1) % 5 == 0:
            run_val(model, val_loader, n_iter, writer)
        torch.save(
            model.module.state_dict(), args.checkpoint_dir / f"epoch{epoch}.pth"
        )


def main() -> None:
    # ngpus = torch.cuda.device_count()
    # torch.multiprocessing.spawn(train, nprocs=ngpus)
    train()


if __name__ == "__main__":
    main()