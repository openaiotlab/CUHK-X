import os
import gc
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ================== setting ==================
DATASET_PATH = "/aiot-nvme-15T-x2-hk01/siyang/CUHK-X-Final/SM_data/RGB"
DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

# ================== global unified action classes ==================
def get_all_action_classes(root_dir):
    actions = set()
    for action in os.listdir(root_dir):
        action_path = os.path.join(root_dir, action)
        if os.path.isdir(action_path):
            actions.add(action)
    return sorted(actions)

ALL_ACTIONS = get_all_action_classes(DATASET_PATH)
assert len(ALL_ACTIONS) == 44, f"Expected 44 actions, got {len(ALL_ACTIONS)}"
print(f"Total actions: {len(ALL_ACTIONS)}")
ACTION_TO_IDX = {action: idx for idx, action in enumerate(ALL_ACTIONS)}  # global mapping

# ================== data augmentation ==================
def get_train_transform(image_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_test_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# ================== Dataset ==================
class ActionDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ================== 数据加载器 ==================
def get_dataloaders(root_dir, test_user, batch_size, image_size, num_users=30):
    """
    Construct data loaders:
    - supervised: other 29 users + all 44 actions (global labels)
    - test: all data of test_user (global labels)
    """
    # === 1. Construct supervised learning data (other users + all 44 actions) ===
    supervised_data = []
    all_users = [f"user{i}" for i in range(1, num_users + 1)]
    other_users = [u for u in all_users if u != test_user]

    for action in ALL_ACTIONS:
        action_path = os.path.join(root_dir, action)
        if not os.path.isdir(action_path):
            continue
        for user in other_users:
            user_action_path = os.path.join(action_path, user)
            if not os.path.isdir(user_action_path):
                continue
            for trial in os.listdir(user_action_path):
                trial_path = os.path.join(user_action_path, trial)
                if os.path.isdir(trial_path):
                    for img_file in os.listdir(trial_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(trial_path, img_file)
                            label = ACTION_TO_IDX[action]  # global label 0-43
                            supervised_data.append((img_path, label))

    # === 2. Construct test data (test_user + all 44 actions) ===
    test_data = []
    for action in ALL_ACTIONS:
        action_path = os.path.join(root_dir, action)
        if not os.path.isdir(action_path):
            continue
        user_action_path = os.path.join(action_path, test_user)
        if not os.path.isdir(user_action_path):
            continue
        for trial in os.listdir(user_action_path):
            trial_path = os.path.join(user_action_path, trial)
            if os.path.isdir(trial_path):
                for img_file in os.listdir(trial_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(trial_path, img_file)
                        label = ACTION_TO_IDX[action]  # global label
                        test_data.append((img_path, label))

    # === 3. Create datasets and DataLoader ===
    train_transform = get_train_transform(image_size)
    test_transform = get_test_transform(image_size)

    supervised_dataset = ActionDataset(supervised_data, transform=train_transform)
    test_dataset = ActionDataset(test_data, transform=test_transform)

    pin_memory = torch.cuda.is_available()

    supervised_loader = DataLoader(
        supervised_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=pin_memory,
        drop_last=True  
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory
    )

    print(f"Supervised  {len(supervised_data)} samples (44 actions, 29 users)")
    print(f"Test  {len(test_data)} samples (test_user, global 44-class labels)")

    return supervised_loader, test_loader

# ================== model ==================
class ActionClassifier(nn.Module):
    def __init__(self, num_classes=44):
        super(ActionClassifier, self).__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        base_model = models.resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        self.classifier = nn.Linear(512, num_classes)  # fixed 44 classes

    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)

# ================== training function ==================
def train_model(supervised_loader, test_loader, epochs, learning_rate):
    model = ActionClassifier(num_classes=44).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        steps = 0

        for images, labels in tqdm(supervised_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            try:
                images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1
            except RuntimeError as e:
                print(f"Warning: Runtime error during training: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                continue

        avg_loss = total_loss / steps if steps > 0 else 0.0
        
        torch.cuda.empty_cache()
        gc.collect()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f} | Test Acc: {acc * 100:.2f}%")

# ================== main program ==================
def parse_args():
    parser = argparse.ArgumentParser(description="Global 44-Class Supervised Cross-user Action Recognition")
    parser.add_argument("--dataset_path", type=str, default="/aiot-nvme-15T-x2-hk01/siyang/CUHK-X-Final/SM_data/RGB")
    parser.add_argument("--test_user", type=str, required=True)
    parser.add_argument("--num_users", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if not args.test_user.startswith("user") or not args.test_user[4:].isdigit():
        raise ValueError("test_user must be in format 'userX'")

    supervised_loader, test_loader = get_dataloaders(
        root_dir=args.dataset_path,
        test_user=args.test_user,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_users=args.num_users
    )

    train_model(
        supervised_loader=supervised_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        learning_rate=args.lr
    )