import os
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse

# ================== setting ==================
DATASET_PATH = "/aiot-nvme-15T-x2-hk01/siyang/CUHK-X-Final/SM_data/RGB"
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

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
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),  # hue=0 to avoid overflow
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_test_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# ================== general Dataset ==================
class ActionDataset(Dataset):
    def __init__(self, data_list, transform=None, is_contrastive=False):
        self.data_list = data_list
        self.transform = transform
        self.is_contrastive = is_contrastive

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_contrastive:
            img_path = self.data_list[idx]
            image = Image.open(img_path).convert("RGB")
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2
        else:
            img_path, label = self.data_list[idx]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label

# ================== data loaders ==================
def get_dataloaders(root_dir, test_user, batch_size, image_size, num_users=30):
    """
    Construct data loaders:
    - supervised: other 29 users + all 44 actions (global labels)
    - contrastive: all data of test_user (no labels)
    - test: all data of test_user (global labels, only actual performed actions)
    """
    # === 1. Construct supervised data (other users + all 44 actions) ===
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

    # === 2. Get all image paths of test_user ===
    contrastive_image_paths = []
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
                        contrastive_image_paths.append(img_path)
                        label = ACTION_TO_IDX[action]  # global label
                        test_data.append((img_path, label))

    # === 3. Create datasets ===
    train_transform = get_train_transform(image_size)
    test_transform = get_test_transform(image_size)

    supervised_dataset = ActionDataset(supervised_data, transform=train_transform, is_contrastive=False)
    contrastive_dataset = ActionDataset(contrastive_image_paths, transform=train_transform, is_contrastive=True)
    test_dataset = ActionDataset(test_data, transform=test_transform, is_contrastive=False)

    # === 4. Create DataLoader (add drop_last to prevent BatchNorm errors) ===
    pin_memory = torch.cuda.is_available()

    supervised_loader = DataLoader(
        supervised_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=pin_memory,
        drop_last=True
    )

    contrastive_loader = DataLoader(
        contrastive_dataset,
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
    print(f"Contrastive  {len(contrastive_image_paths)} samples (test_user all data)")
    print(f"Test  {len(test_data)} samples (test_user, global 44-class labels)")

    return supervised_loader, contrastive_loader, test_loader

# ================== model ==================
class UnifiedModel(nn.Module):
    def __init__(self, base_model, num_classes=44):
        super(UnifiedModel, self).__init__()
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.classification_head = nn.Linear(512, num_classes)  # fixed 44 classes

    def forward(self, x, contrastive=False):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        if contrastive:
            return self.projection_head(features)
        else:
            return self.classification_head(features)

# ================== contrastive loss ==================
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, proj1, proj2):
        proj1 = nn.functional.normalize(proj1, dim=1)
        proj2 = nn.functional.normalize(proj2, dim=1)
        sim = torch.matmul(proj1, proj2.T) / self.temperature
        labels = torch.arange(sim.size(0)).to(sim.device)
        return nn.CrossEntropyLoss()(sim, labels)

# ================== training function ==================
def train_model(supervised_loader, contrastive_loader, test_loader, epochs, learning_rate, temperature):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    resnet_base = models.resnet18(weights=weights)
    model = UnifiedModel(resnet_base, num_classes=44).to(DEVICE)  # fixed 44 classes
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    con_criterion = ContrastiveLoss(temperature=temperature)
    sup_criterion = nn.CrossEntropyLoss()

    sup_iter = iter(supervised_loader)
    con_iter = iter(contrastive_loader)

    for epoch in range(epochs):
        model.train()
        total_sup_loss = 0.0
        total_con_loss = 0.0
        sup_steps = 0
        con_steps = 0

        max_steps = max(len(supervised_loader), len(contrastive_loader))

        for _ in tqdm(range(max_steps), desc=f"Epoch {epoch + 1}/{epochs}"):
            # Supervised learning
            try:
                images, labels = next(sup_iter)
            except StopIteration:
                sup_iter = iter(supervised_loader)
                images, labels = next(sup_iter)
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            # Contrastive learning
            try:
                img1, img2 = next(con_iter)
            except StopIteration:
                con_iter = iter(contrastive_loader)
                img1, img2 = next(con_iter)
            img1, img2 = img1.to(DEVICE, non_blocking=True), img2.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            # Contrastive loss
            p1 = model(img1, contrastive=True)
            p2 = model(img2, contrastive=True)
            con_loss = con_criterion(p1, p2)

            # Supervised loss
            out = model(images, contrastive=False)
            sup_loss = sup_criterion(out, labels)

            total_loss = con_loss + sup_loss
            total_loss.backward()
            optimizer.step()

            total_sup_loss += sup_loss.item()
            total_con_loss += con_loss.item()
            sup_steps += 1
            con_steps += 1

        avg_sup = total_sup_loss / sup_steps
        avg_con = total_con_loss / con_steps

        # Testing (44 classes)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                outputs = model(images, contrastive=False)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch + 1}/{epochs} | Sup Loss: {avg_sup:.4f} | Con Loss: {avg_con:.4f} | Test Acc: {acc * 100:.2f}%")

# ================== main function ==================
def parse_args():
    parser = argparse.ArgumentParser(description="Global 44-Class Cross-user Action Recognition with SimCLR")
    parser.add_argument("--dataset_path", type=str, default="/aiot-nvme-15T-x2-hk01/siyang/CUHK-X-Final/SM_data/RGB")
    parser.add_argument("--test_user", type=str, required=True)
    parser.add_argument("--num_users", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--image_size", type=int, default=224)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if not args.test_user.startswith("user") or not args.test_user[4:].isdigit():
        raise ValueError("test_user must be in format 'userX'")

    supervised_loader, contrastive_loader, test_loader = get_dataloaders(
        root_dir=args.dataset_path,
        test_user=args.test_user,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_users=args.num_users
    )

    train_model(
        supervised_loader=supervised_loader,
        contrastive_loader=contrastive_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        temperature=args.temperature
    )