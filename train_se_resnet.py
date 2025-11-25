import os
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from monai.networks.nets import SEResNet50

from dataset import (
    load_yaml_list,
    VertebraSagittalDataset,
    binary_label_from_filename,
)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_train_val_split(train_files, val_fraction=0.1, seed=42):
    rng = np.random.RandomState(seed)
    indices = np.arange(len(train_files))
    rng.shuffle(indices)

    n_val = int(len(indices) * val_fraction)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_split = [train_files[i] for i in train_idx]
    val_split = [train_files[i] for i in val_idx]

    return train_split, val_split

def compute_class_weights(file_list):
    labels = [binary_label_from_filename(f) for f in file_list]
    counts = np.bincount(labels, minlength=2)
    counts = np.maximum(counts, 1)  # avoid zero
    weights = 1.0 / counts
    weights = weights * (2.0 / weights.sum())
    return torch.tensor(weights, dtype=torch.float32)

def evaluate(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = float("nan")

    return {
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "y_true": all_labels,
        "y_pred": all_preds,
        "y_prob": all_probs,
    }

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    img_root = Path(args.img_root)

    train_files_full = load_yaml_list(args.train_yaml)
    test_files = load_yaml_list(args.test_yaml)

    train_files, val_files = make_train_val_split(
        train_files_full, val_fraction=args.val_fraction, seed=args.seed
    )

    print(
        f"Train: {len(train_files)},  Val: {len(val_files)},  Test: {len(test_files)}"
    )

    
    train_ds = VertebraSagittalDataset(img_root, train_files, resize=args.img_size)
    val_ds = VertebraSagittalDataset(img_root, val_files, resize=args.img_size)
    test_ds = VertebraSagittalDataset(img_root, test_files, resize=args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SEResNet50(
        spatial_dims=2,
        in_channels=1,
        num_classes=2,
    ).to(device)

    class_weights = compute_class_weights(train_files).to(device)
    print(f"\nClass weights: {class_weights.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    best_val_auc = -1
    best_epoch = -1
    os.makedirs(args.out_dir, exist_ok=True)
    pt_path = Path(args.out_dir) / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0
        batches = 0

        for imgs, labels, _ in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        train_loss = running_loss / batches

        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["auc"])

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f}"
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_auc": best_val_auc,
                    "args": vars(args),
                },
                pt_path,
            )

            tar_path = Path(args.out_dir) / f"best_ckpt_epoch{epoch}_{best_val_auc:.4f}.tar"
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),  
                    "optimizer": optimizer.state_dict(),
                    "val_auc": best_val_auc,
                },
                tar_path,
            )

            print(f"  --> Best model updated. Saved .pt and .tar\n")

    print(f"\nTraining done. Best epoch = {best_epoch} with AUC = {best_val_auc:.4f}")
    print(f"Best model (.pt): {pt_path}")

    print("\n=== TEST EVALUATION ===")

    checkpoint = torch.load(pt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_metrics = evaluate(model, test_loader, device)

    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1:       {test_metrics['f1']:.4f}")
    print(f"Test AUC:      {test_metrics['auc']:.4f}")

    print("\nClassification Report:")
    print(classification_report(test_metrics["y_true"], test_metrics["y_pred"], digits=4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_root", type=str, default=r"D:\Binary Classification\Data\img")
    parser.add_argument("--train_yaml", type=str, default="train_file_list.delx.yaml")
    parser.add_argument("--test_yaml", type=str, default="test_file_list.delx.yaml")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
