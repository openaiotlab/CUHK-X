import argparse
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


# ---------- Arg parsing helpers ----------
def parse_labels_arg(s: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Parse --labels argument.
    Accepts: "10,30", "10:30", "[10,30]", "(10,30)". Returns (start,end) 1-based inclusive.
    '0' or 'all' or None -> return None (use all labels).
    """
    if s is None:
        return None
    s = str(s).strip().lower()
    if s in ("0", "all", ""):
        return None
    s = s.strip("[]() ").replace(" ", "").replace(":", ",")
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "--labels expects a range like '10,30' or '[10,30]' or '10:30'"
        )
    try:
        a, b = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Both start and end in --labels must be integers."
        ) from exc
    if a <= 0 or b <= 0:
        raise argparse.ArgumentTypeError("--labels ranks are 1-based and must be > 0.")
    if a > b:
        a, b = b, a
    return a, b


# ---------- Logging ----------
def configure_logging(log_dir: Path, args: argparse.Namespace) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    labels_range = parse_labels_arg(args.labels)
    labels_tag = (
        f"labels{labels_range[0]}-{labels_range[1]}" if labels_range else "labelsAll"
    )
    log_filename = (
        log_dir
        / f"training_{args.data}_{args.network}_{args.split_mode}_{args.weights}_{labels_tag}_{timestamp}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return log_filename


# ---------- Data loading ----------
def load_data(
    root_dir: Path,
    split_mode: str,
    test_user_id: Optional[int] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    label_rank_range: Optional[Tuple[int, int]] = None,
) -> Tuple[List[Tuple[str, int, int]], List[Tuple[str, int, int]], dict]:
    """
    Loads and splits image data based on the specified mode. Supports .jpg and .png.
    Optional label_rank_range=(start,end) selects labels by frequency rank BEFORE splitting.
    Returns:
        train_data: List of (image_path, mapped_label, user_id)
        test_data:  List of (image_path, mapped_label, user_id)
        index_to_label: dict {mapped_index: original_label}
    """
    all_data: List[Tuple[str, int, int]] = []
    for label_dir in os.listdir(root_dir):
        label_path = root_dir / label_dir
        if not label_path.is_dir():
            continue
        label_matches = re.findall(r"\d+", label_dir)
        if not label_matches:
            continue
        label = int(label_matches[0])

        for user_dir in os.listdir(label_path):
            user_path = label_path / user_dir
            if not user_path.is_dir():
                continue
            user_matches = re.findall(r"\d+", user_dir)
            if not user_matches:
                continue
            user_id = int(user_matches[0])

            for sequence_dir in os.listdir(user_path):
                sequence_path = user_path / sequence_dir
                if not sequence_path.is_dir():
                    continue
                for image_file in os.listdir(sequence_path):
                    if image_file.lower().endswith((".jpg", ".png")) and user_id <= 30:
                        image_path = sequence_path / image_file
                        all_data.append((str(image_path), label, user_id))

    if not all_data:
        raise ValueError(f"No images found under root_dir: {root_dir}")

    full_label_counts = Counter(item[1] for item in all_data)
    logging.info(
        "Full dataset label distribution (before filtering): %s", full_label_counts
    )

    freq_sorted_labels: List[int] = [lbl for lbl, _ in full_label_counts.most_common()]

    if label_rank_range:
        start, end = label_rank_range
        n = len(freq_sorted_labels)
        start = max(1, start)
        end = min(end, n)
        if start > end:
            raise ValueError(f"Invalid --labels range after clamping: {start}>{end}")
        selected_labels = freq_sorted_labels[start - 1 : end]
        selected_set = set(selected_labels)
        filtered_data = [item for item in all_data if item[1] in selected_set]
        selected_counts = {lbl: full_label_counts[lbl] for lbl in selected_labels}
        logging.info(
            "Using labels by frequency rank %d-%d (1=highest): %s",
            start,
            end,
            selected_counts,
        )
        mapping_order = selected_labels
    else:
        filtered_data = all_data
        logging.info("Using all %d labels.", len(full_label_counts))
        mapping_order = sorted(full_label_counts.keys())

    if not filtered_data:
        raise ValueError(
            "After applying label frequency range filtering, no data remains."
        )

    label_to_index = {orig: i for i, orig in enumerate(mapping_order)}
    index_to_label = {i: orig for orig, i in label_to_index.items()}
    logging.info("Label mapping (original -> new index): %s", label_to_index)

    mapped_data = [
        (path, label_to_index[label], user_id)
        for (path, label, user_id) in filtered_data
    ]

    if split_mode == "cross":
        if test_user_id is None:
            raise ValueError("test_user_id must be provided for 'cross' mode.")
        train_data = [item for item in mapped_data if item[2] != test_user_id]
        test_data = [item for item in mapped_data if item[2] == test_user_id]
    elif split_mode == "intra":
        labels_for_stratify = [item[1] for item in mapped_data]
        train_data, test_data = train_test_split(
            mapped_data,
            test_size=test_size,
            random_state=random_state,
            stratify=labels_for_stratify,
        )
    else:
        raise ValueError(f"Invalid split_mode: {split_mode}")

    if not train_data or not test_data:
        logging.warning(
            "After filtering/splitting, train or test set is empty. "
            "Check your --labels and split settings."
        )

    return train_data, test_data, index_to_label


# ---------- Oversampling ----------
def perform_oversampling(
    train_data: Sequence[Tuple[str, int, int]]
) -> List[Tuple[str, int, int]]:
    class_counts = Counter(item[1] for item in train_data)
    if not class_counts:
        return list(train_data)

    logging.info("Original training class distribution: %s", class_counts)

    majority_class_count = max(class_counts.values())
    sorted_classes = class_counts.most_common()
    minority_classes_to_oversample = {label for label, _ in sorted_classes[-5:]}
    for label, count in class_counts.items():
        if count < 100:
            minority_classes_to_oversample.add(label)

    logging.info(
        "Identified minority classes for oversampling: %s",
        minority_classes_to_oversample,
    )

    oversampled_data = list(train_data)
    for label in minority_classes_to_oversample:
        class_samples = [item for item in train_data if item[1] == label]
        current_count = len(class_samples)
        samples_to_add = majority_class_count - current_count

        if samples_to_add > 0 and class_samples:
            oversampled_data.extend(random.choices(class_samples, k=samples_to_add))

    logging.info(
        "New training class distribution: %s",
        Counter(item[1] for item in oversampled_data),
    )
    return oversampled_data


# ---------- Dataset ----------
class ImageDataset(Dataset):
    def __init__(self, data: Sequence[Tuple[str, int, int]], transform=None):
        self.data = list(data)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        image_path, label, _ = self.data[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)
        except (IOError, UnidentifiedImageError, OSError) as exc:
            print(f"Skipping corrupted image: {image_path} due to {exc}")
            return None, None


def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)


# ---------- Main training loop ----------
def main():
    parser = argparse.ArgumentParser(description="Train a model on image data.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        help="root data dir, contains RGB、Depth_Color、IR、Thermal ",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="rgb",
        choices=["rgb", "depth", "ir", "thermal"],
        help="Data modality to use.",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device number to use.")
    parser.add_argument(
        "--network",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50", "vit_b_16"],
        help="Network architecture.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="pretrained",
        choices=["pretrained", "scratch"],
        help="Weight initialization: 'pretrained' or 'scratch'.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="cross",
        choices=["cross", "intra"],
        help="Data splitting mode: 'cross' (by user_id) or 'intra' (random 80/20 split).",
    )
    parser.add_argument(
        "--oversample",
        action="store_true",
        help="Enable minority class oversampling in the training set.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Label frequency rank range, e.g., '10,30' or '[1,5]'. 1-based inclusive. Use '0' or 'all' for all labels.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        help="log output directory.",
    )
    parser.add_argument(
        "--cross_user_id",
        type=int,
        default=5,
        help="under cross mode the testing user ID。",
    )
    args = parser.parse_args()
    labels_range = parse_labels_arg(args.labels)

    log_dir = Path(args.log_dir)
    log_file = configure_logging(log_dir, args)
    logging.info("Log file: %s", log_file)
    logging.info("Starting training with arguments: %s", args)
    logging.info(
        "Parsed labels range: %s",
        labels_range if labels_range else "All",
    )

    modality_subdirs = {
        "rgb": "RGB",
        "depth": "Depth_Color",
        "ir": "IR",
        "thermal": "Thermal",
    }
    dataset_root = Path(args.dataset_root)
    modality_dirname = modality_subdirs[args.data]
    root_dir = dataset_root / modality_dirname

    if not root_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {root_dir}")

    train_data, test_data, index_to_label = load_data(
        root_dir=root_dir,
        split_mode=args.split_mode,
        test_user_id=args.cross_user_id,
        label_rank_range=labels_range,
    )

    if args.oversample:
        logging.info("Oversampling enabled. Analyzing training data...")
        train_data = perform_oversampling(train_data)

    image_size = 224
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    train_dataset = ImageDataset(train_data, transform=data_transforms["train"])
    test_dataset = ImageDataset(test_data, transform=data_transforms["test"])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    logging.info(
        "Training with %s (%s) on %s data, split_mode: %s.",
        args.network,
        args.weights,
        args.data,
        args.split_mode,
    )
    logging.info(
        "Training samples: %d, Testing samples: %d",
        len(train_dataset),
        len(test_dataset),
    )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    num_classes = len(index_to_label)
    weights_param = None
    if args.weights == "pretrained":
        if args.network == "resnet18":
            weights_param = models.ResNet18_Weights.IMAGENET1K_V1
        elif args.network == "resnet34":
            weights_param = models.ResNet34_Weights.IMAGENET1K_V1
        elif args.network == "resnet50":
            weights_param = models.ResNet50_Weights.IMAGENET1K_V1
        elif args.network == "vit_b_16":
            weights_param = models.ViT_B_16_Weights.IMAGENET1K_V1

    if args.network.startswith("resnet"):
        if args.network == "resnet18":
            model = models.resnet18(weights=weights_param)
        elif args.network == "resnet34":
            model = models.resnet34(weights=weights_param)
        elif args.network == "resnet50":
            model = models.resnet50(weights=weights_param)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.network == "vit_b_16":
        model = models.vit_b_16(weights=weights_param)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Invalid network architecture: {args.network}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        train_loop = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False
        )
        for inputs, labels in train_loop:
            if inputs.nelement() == 0:
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loop.set_postfix(loss=loss.item())

        model.eval()
        all_labels, all_predictions = [], []
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                if inputs.nelement() == 0:
                    continue
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
        precision = precision_score(
            all_labels, all_predictions, average="macro", zero_division=0
        )
        recall = recall_score(
            all_labels, all_predictions, average="macro", zero_division=0
        )

        log_msg = (
            f"Epoch {epoch + 1}/{args.epochs} -> Test Accuracy: {accuracy:.4f}, "
            f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
        )
        print(log_msg)
        logging.info(log_msg)

    logging.info("Finished Training. Calculating final per-class accuracy...")

    unique_labels_in_test = np.unique(all_labels)
    if len(unique_labels_in_test) > 0:
        cm = confusion_matrix(
            all_labels, all_predictions, labels=unique_labels_in_test
        )
        if cm.shape[0] > 0:
            class_accuracies = cm.diagonal() / cm.sum(axis=1)
            per_class_log = ["\n--- Final Per-Class Test Accuracy (by original label) ---"]
            for i, mapped_label in enumerate(unique_labels_in_test):
                acc = class_accuracies[i]
                correct = cm.diagonal()[i]
                total = cm.sum(axis=1)[i]
                original_label = index_to_label.get(int(mapped_label), int(mapped_label))
                per_class_log.append(
                    f"Class {original_label} (mapped {int(mapped_label)}): {acc:.4f} "
                    f"({correct}/{total} correct)"
                )
            final_log_message = "\n".join(per_class_log)
            print(final_log_message)
            logging.info(final_log_message)
        else:
            logging.info("Confusion matrix was empty; could not calculate per-class accuracy.")
    else:
        logging.info("No labels found in the test set; skipping per-class accuracy calculation.")


if __name__ == "__main__":
    main()