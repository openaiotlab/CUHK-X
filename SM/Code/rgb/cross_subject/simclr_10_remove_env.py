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
import re

# ================== setting ==================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== enhanced data augmentation ==================
def get_train_transform(image_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomApply([
        #     transforms.RandomRotation(degrees=15)  # 1. Rotation
        # ], p=0.3),
        
        # transforms.RandomApply([
        #     transforms.RandomPerspective(distortion_scale=0.2, p=0.3)  # 2. Perspective
        # ], p=0.3),
        
        # transforms.RandomApply([
        #     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10)  # 3. Affine (translation + shear)
        # ], p=0.3),
        
        # transforms.RandomApply([
        #     transforms.RandomSolarize(threshold=128)  # 4. Solarize
        # ], p=0.2),
        
        # transforms.RandomApply([
        #     transforms.RandomPosterize(bits=4)  # 5. Posterize
        # ], p=0.2),
        
        # transforms.RandomApply([
        #     transforms.RandomAdjustSharpness(sharpness_factor=2)  # 6. Sharpness
        # ], p=0.3),
        
        # brightness, contrast, saturation remain unchanged
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=int(0.1 * image_size) // 2 * 2 + 1, sigma=(0.1, 2.0)),
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

# ================== get actions actually performed by user ==================
def get_user_executed_actions(root_dir, user):
    executed_actions = set()
    for action in os.listdir(root_dir):
        action_path = os.path.join(root_dir, action)
        if not os.path.isdir(action_path):
            continue
        user_action_path = os.path.join(action_path, user)
        if os.path.isdir(user_action_path):
            has_data = False
            for trial in os.listdir(user_action_path):
                trial_path = os.path.join(user_action_path, trial)
                if os.path.isdir(trial_path):
                    for img_file in os.listdir(trial_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            has_data = True
                            break
                    if has_data:
                        break
            if has_data:
                executed_actions.add(action)
    return sorted(executed_actions)

# ================== environment-aware user partition ==================
def get_environment_users(test_user, num_users=30):
    """
    Return other users in the same environment as test_user
    - Environment A: user1 ~ user15
    - Environment B: user16 ~ user30
    """
    try:
        test_idx = int(test_user[4:])
    except (ValueError, IndexError):
        raise ValueError(f"Invalid test_user format: {test_user}")
    
    if test_idx < 1 or test_idx > num_users:
        raise ValueError(f"User index {test_idx} out of range [1, {num_users}]")
    
    if test_idx <= 15:
        # Environment A: user1 ~ min(15, num_users)
        env_start, env_end = 1, min(15, num_users)
        env_name = "A (users 1-15)"
    else:
        # Environment B: max(16, ...) ~ num_users
        env_start, env_end = 16, num_users
        env_name = f"B (users 16-{num_users})"
    
    all_env_users = [f"user{i}" for i in range(env_start, env_end + 1)]
    other_users = [u for u in all_env_users if u != test_user]
    
    print(f"{test_user} assigned to environment {env_name}. "
          f"Supervised training uses {len(other_users)} users: {other_users}")
    return other_users

# ================== data loaders ==================
def get_dataloaders(root_dir, test_user, batch_size, image_size, num_users=30, supervised_action_names=None):
    # === 1. Get actions actually performed by test_user ===
    test_user_executed = get_user_executed_actions(root_dir, test_user)
    print(f"Target user {test_user} executed {len(test_user_executed)} actions: {test_user_executed}")
    
    # === 2. Determine supervised action list: intersection calculation ===
    if supervised_action_names is not None:
        supervised_set = set(supervised_action_names)
        executed_set = set(test_user_executed)
        intersection_actions = sorted(list(supervised_set & executed_set))
        if len(intersection_actions) == 0:
            raise ValueError(f"No overlap between supervised actions and test_user {test_user} executed actions")
        print(f"Supervised actions intersection: {intersection_actions} (size: {len(intersection_actions)})")
        actions_for_sup = intersection_actions
    else:
        actions_for_sup = test_user_executed
        print(f"Using all executed actions by {test_user} for supervised training: {actions_for_sup}")

    local_action_to_idx = {action: idx for idx, action in enumerate(actions_for_sup)}

    # === 3. Get other users in the same environment (key modification) ===
    other_users = get_environment_users(test_user, num_users)

    # === 4. Construct supervised learning data (only same environment users) ===
    supervised_data = []
    for action in actions_for_sup:
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
                            label = local_action_to_idx[action]
                            supervised_data.append((img_path, label))

    # === 5. Construct contrastive and test data (unchanged) ===
    contrastive_image_paths = []
    test_data = []

    for action in test_user_executed:
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
                        if action in local_action_to_idx:
                            label = local_action_to_idx[action]
                            test_data.append((img_path, label))

    # === 6. Create datasets and DataLoaders ===
    train_transform = get_train_transform(image_size)
    test_transform = get_test_transform(image_size)

    supervised_dataset = ActionDataset(supervised_data, transform=train_transform, is_contrastive=False) if supervised_data else None
    contrastive_dataset = ActionDataset(contrastive_image_paths, transform=train_transform, is_contrastive=True)
    test_dataset = ActionDataset(test_data, transform=test_transform, is_contrastive=False)

    pin_memory = torch.cuda.is_available()

    supervised_loader = None
    if supervised_dataset and len(supervised_dataset) > 0:
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

    print(f"Supervised  {len(supervised_data)} samples (from {len(actions_for_sup)} actions)")
    print(f"Contrastive  {len(contrastive_image_paths)} samples")
    print(f"Test  {len(test_data)} samples")

    return supervised_loader, contrastive_loader, test_loader, len(actions_for_sup)

# ================== model ==================
class UnifiedModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super(UnifiedModel, self).__init__()
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        self.projection_head = nn.Sequential(
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 128),
            nn.BatchNorm1d(128, affine=False)
        )
        self.classification_head = nn.Linear(512, num_classes)

    def forward(self, x, contrastive=False):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        if contrastive:
            return self.projection_head(features)
        else:
            return self.classification_head(features)

# ================== contrastive loss ==================
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T) / self.temperature
        N = z1.size(0)
        labels = torch.arange(N, device=z.device)
        labels = torch.cat([labels + N, labels], dim=0)
        mask = (~torch.eye(2 * N, dtype=torch.bool, device=z.device)).float()
        sim = sim * mask + (1 - mask) * (-9e15)
        loss = nn.CrossEntropyLoss()(sim, labels)
        return loss

# ================== training function ==================
def train_model(supervised_loader, contrastive_loader, test_loader, epochs, learning_rate, num_classes, temperature):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    resnet_base = models.resnet18(weights=weights)
    model = UnifiedModel(resnet_base, num_classes=num_classes).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    con_criterion = NTXentLoss(temperature=temperature)
    sup_criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()

    sup_weight = 1.0
    con_weight = 0.5

    has_supervised = supervised_loader is not None
    if has_supervised:
        sup_iter = iter(supervised_loader)
    con_iter = iter(contrastive_loader)

    for epoch in range(epochs):
        model.train()
        total_sup_loss = 0.0
        total_con_loss = 0.0
        sup_steps = 0
        con_steps = 0

        max_steps = len(contrastive_loader)
        if has_supervised:
            max_steps = max(max_steps, len(supervised_loader))

        for _ in tqdm(range(max_steps), desc=f"Epoch {epoch + 1}/{epochs}"):
            sup_loss = torch.tensor(0.0, device=DEVICE)
            if has_supervised:
                try:
                    images, labels = next(sup_iter)
                except StopIteration:
                    sup_iter = iter(supervised_loader)
                    images, labels = next(sup_iter)
                images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            try:
                img1, img2 = next(con_iter)
            except StopIteration:
                con_iter = iter(contrastive_loader)
                img1, img2 = next(con_iter)
            img1, img2 = img1.to(DEVICE, non_blocking=True), img2.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                p1 = model(img1, contrastive=True)
                p2 = model(img2, contrastive=True)
                con_loss = con_criterion(p1, p2)
                total_con_loss += con_loss.item()
                con_steps += 1

                if has_supervised:
                    out = model(images, contrastive=False)
                    sup_loss = sup_criterion(out, labels)
                    total_sup_loss += sup_loss.item()
                    sup_steps += 1

                loss = con_weight * con_loss + sup_weight * sup_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        avg_sup = total_sup_loss / sup_steps if sup_steps > 0 else 0.0
        avg_con = total_con_loss / con_steps if con_steps > 0 else 0.0

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
    parser = argparse.ArgumentParser(description="Environment-aware Cross-user Action Recognition")
    parser.add_argument("--dataset_path", type=str, default="/aiot-nvme-15T-x2-hk01/siyang/CUHK-X-Final/SM_data/RGB")
    parser.add_argument("--test_user", type=str, required=True)
    parser.add_argument("--num_users", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--supervised_action_ids", type=int, nargs='+', default=None,
                        help="Action prefixes for supervised training, e.g., 6 7 9 11 12 20 21 32 36 37")
    parser.add_argument("--exact_match", action="store_true", help="Exact prefix match (no trailing digits)")
    return parser.parse_args()

def get_all_action_classes(root_dir):
    actions = set()
    for action in os.listdir(root_dir):
        action_path = os.path.join(root_dir, action)
        if os.path.isdir(action_path):
            actions.add(action)
    return sorted(actions)

if __name__ == "__main__":
    args = parse_args()

    if not args.test_user.startswith("user") or not args.test_user[4:].isdigit():
        raise ValueError("test_user must be in format 'userX'")

    ALL_ACTIONS = get_all_action_classes(args.dataset_path)
    print(f"Total actions in dataset: {len(ALL_ACTIONS)}")

    supervised_action_names = None
    if args.supervised_action_ids is not None:
        supervised_action_names = []
        for prefix in args.supervised_action_ids:
            if args.exact_match:
                matched = [action for action in ALL_ACTIONS if re.match(rf"^{prefix}(?!\d)", action)]
            else:
                matched = [action for action in ALL_ACTIONS if action.startswith(str(prefix))]
            if not matched:
                print(f"Warning: No action starts with '{prefix}'")
            else:
                supervised_action_names.extend(matched)
        supervised_action_names = sorted(set(supervised_action_names))
        print(f"Supervised training on actions: {supervised_action_names}")

    supervised_loader, contrastive_loader, test_loader, num_classes = get_dataloaders(
        root_dir=args.dataset_path,
        test_user=args.test_user,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_users=args.num_users,
        supervised_action_names=supervised_action_names
    )

    print(f"Building {num_classes}-class classifier")
    train_model(
        supervised_loader=supervised_loader,
        contrastive_loader=contrastive_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        num_classes=num_classes,
        temperature=args.temperature
    )