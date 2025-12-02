import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import logging
import random
import pandas as pd
from scipy.spatial.distance import cdist
from datetime import datetime

# ----------------------
# system config module
# ----------------------
# setting random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# parameter configuration
config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "window_size": 20,        # time window length (frames)
    "max_points": 112,        # max points per frame
    "batch_size": 32,         # batch size
    "num_epochs": 40,         # number of epochs
    "learning_rate": 1e-3,    # learning rate
    "num_workers": 8,         # number of data loading workers
    # Legacy single path support (for backward compatibility)
    "data_root": "/aiot-nvme-15T-x2-hk01/siyang/CUHK-X-example/SM/Data",  
    "log_dir": "/aiot-nvme-15T-x2-hk01/siyang/CUHK-X-example/SM/Code/radar/logs_radar_all"
}

# initialize logging system
os.makedirs(config["log_dir"], exist_ok=True)
logging.basicConfig(
    filename=os.path.join(config["log_dir"], 'training_evaluation_mix_radar_log.txt'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info(f"Training Config: {config}")

# ----------------------
# data preprocessing module
# ----------------------
class LabelProcessor:
    """Label processing and validation module - supports multiple data roots"""
    def __init__(self, data_roots):
        """
        Initialize LabelProcessor with single or multiple data roots.
        
        Args:
            data_roots: str or list of str - single path or list of dataset paths
        """
        # Support both single path (backward compatible) and multiple paths
        if isinstance(data_roots, str):
            self.data_roots = [data_roots]
        else:
            self.data_roots = list(data_roots)
        
        self.valid_labels = {}
        self._process_labels()
    
    def _process_labels(self):
        """Scan all data directories and build unified valid label mapping"""
        import re
        
        # Process each data root
        for data_root in self.data_roots:
            radar_root = os.path.join(data_root, 'Radar')
            if not os.path.exists(radar_root):
                logging.warning(f"Radar directory not found: {radar_root}")
                continue
            
            logging.info(f"Processing data root: {radar_root}")
            print(f"Processing data root: {radar_root}")
                
            for folder in os.listdir(radar_root):
                path = os.path.join(radar_root, folder)
                if not os.path.isdir(path):
                    continue
                
                # Parse label information (format: <number><Chinese name>, e.g., 0洗脸, 10搅拌饮料)
                # Use regex to match the leading number
                match = re.match(r'^(\d+)(.*)', folder)
                if not match:
                    continue
                try:
                    label_id = int(match.group(1))
                    label_name = match.group(2) if match.group(2) else f"action_{label_id}"
                except ValueError:
                    continue
                
                # Count total CSV files under all users for this action
                csv_count = 0
                for user_dir in os.listdir(path):
                    user_path = os.path.join(path, user_dir)
                    if not os.path.isdir(user_path):
                        continue
                    for trail_dir in os.listdir(user_path):
                        trail_path = os.path.join(user_path, trail_dir)
                        if not os.path.isdir(trail_path):
                            continue
                        csv_files = [f for f in os.listdir(trail_path) if f.endswith('.csv')]
                        csv_count += len(csv_files)
                
                if csv_count > 0:
                    # If label already exists from another data root, merge the counts and paths
                    if label_id in self.valid_labels:
                        self.valid_labels[label_id]['count'] += csv_count
                        # Store multiple paths for the same label
                        if 'paths' not in self.valid_labels[label_id]:
                            self.valid_labels[label_id]['paths'] = [self.valid_labels[label_id]['path']]
                        self.valid_labels[label_id]['paths'].append(path)
                    else:
                        self.valid_labels[label_id] = {
                            'name': label_name,
                            'path': path,  # primary path (first encountered)
                            'paths': [path],  # all paths for this label
                            'count': csv_count
                        }
        
        # Check if any valid labels were found
        if not self.valid_labels:
            logging.warning("No valid labels found in any data root!")
            self.label_map = {}
            self.num_classes = 0
            return
        
        # Generate continuous label mapping
        sorted_ids = sorted(self.valid_labels.keys())
        self.label_map = {old: new for new, old in enumerate(sorted_ids)}
        self.num_classes = len(sorted_ids)
        
        # Log summary
        total_files = sum(info['count'] for info in self.valid_labels.values())
        logging.info(f"Total labels: {self.num_classes}, Total files: {total_files}")
        print(f"\n=== Label Processing Summary ===")
        print(f"Data roots processed: {len(self.data_roots)}")
        print(f"Total labels: {self.num_classes}")
        print(f"Total files: {total_files}")
        
        # Logging
        # logging.info(f"Valid Labels ({self.num_classes} classes):")
        # for old_id, new_id in self.label_map.items():
        #     info = self.valid_labels[old_id]
        #     logging.info(f"  {old_id} -> {new_id}: {info['name']} (files: {info['count']})")

# ----------------------
# dataset module
# ----------------------
class RadarSequenceDataset(Dataset):
    def __init__(self, label_processor, window_size=20, max_points=128):
        self.window_size = window_size
        self.max_points = max_points
        self.sequences = []
        self.debug_log = []  # Added debug log

        # Iterate over all valid labels
        for label_id, new_id in label_processor.label_map.items():
            # Get all paths for this label (supports multiple data roots)
            label_info = label_processor.valid_labels[label_id]
            action_paths = label_info.get('paths', [label_info['path']])
            
            # Process each action path (may have multiple from different data roots)
            for action_path in action_paths:
                if not os.path.exists(action_path):
                    self._log_debug(f"Path does not exist: {action_path}", level="WARNING")
                    continue
                    
                # Iterate over user folders
                for user_dir in os.listdir(action_path):
                    user_path = os.path.join(action_path, user_dir)
                    if not os.path.isdir(user_path):
                        continue
                        
                    # Iterate over trail folders
                    for trail_dir in os.listdir(user_path):
                        trail_path = os.path.join(user_path, trail_dir)
                        if not os.path.isdir(trail_path):
                            continue
                            
                        # Process each CSV file
                        for csv_file in os.listdir(trail_path):
                            if not csv_file.endswith('.csv'):
                                continue
                            
                            file_path = os.path.join(trail_path, csv_file)
                file_log = {
                    "file": file_path,
                    "total_frames": 0,
                    "valid_windows": 0,
                    "reject_reasons": []
                }
                
                try:
                    # Read full CSV data (performance optimized)
                    df = pd.read_csv(file_path, usecols=['frame','x','y','z','v'])
                    if df.empty:
                        file_log["reject_reasons"].append("Empty data file")
                        continue
                        
                    df['frame'] = df['frame'].astype(int)
                    frames = df['frame'].unique()
                    file_log["total_frames"] = len(frames)
                    
                    # Loose window generation strategy
                    min_frames = max(5, window_size//2)  # Minimum accepted frames
                    if len(frames) < min_frames:
                        file_log["reject_reasons"].append(f"Insufficient frames ({len(frames)} < {min_frames})")
                        continue
                    
                    # Dynamic window step size adjustment
                    step_size = max(1, window_size//4)  # Denser window sampling
                    window_count = 0
                    
                    # Generate windows by frames (allow incomplete windows at the end)
                    for start_idx in range(0, len(frames), step_size):
                        end_idx = start_idx + window_size
                        window_frames = frames[start_idx:end_idx]
                        
                        # Accept partial windows (at least 50% frames)
                        if len(window_frames) < window_size//2:
                            break
                            
                        window_data = [df[df['frame']==f] for f in window_frames]
                        tracked_points = self._track_points(window_data)
                        
                        if tracked_points is not None:
                            self.sequences.append((tracked_points, new_id))
                            window_count += 1
                    
                    file_log["valid_windows"] = window_count
                    self.debug_log.append(file_log)
                    
                except Exception as e:
                    file_log["reject_reasons"].append(f"异常: {str(e)}")
                    self.debug_log.append(file_log)
                    continue

        # Print debug summary
        self._print_debug_summary()
    
    def _log_debug(self, message, level="INFO"):
        """Log debug information"""
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "level": level,
            "message": message
        }
        self.debug_log.append(log_entry)
    
    def _print_debug_summary(self):
        """Print data processing summary"""
        total_files = len(self.debug_log)
        valid_files = sum(1 for log in self.debug_log if log.get('valid_windows',0) > 0)
        if total_files > 0:
            avg_windows = np.mean([log.get('valid_windows',0) for log in self.debug_log])
            valid_pct = valid_files/total_files
        else:
            avg_windows = 0
            valid_pct = 0
        
        print(f"\n=== Data Loading Debug Summary ===")
        print(f"Total files processed: {total_files}")
        print(f"Valid files: {valid_files} ({valid_pct:.1%})")
        print(f"Average windows per file: {avg_windows:.1f}")
        print(f"Total valid sequences: {len(self.sequences)}")
        
        # Print typical error cases
        print("\nTypical error cases:")
        for log in self.debug_log[-5:]:
            if log.get('reject_reasons'):
                print(f"File: {log['file']}")
                print(f"  Reject reasons: {', '.join(log['reject_reasons'])}")
    
    def _track_points(self, window_data):
        """Loose point tracking (allow partial loss)"""
        try:
            tracks = {}
            for frame_idx, frame_data in enumerate(window_data):
                current_points = frame_data[['x','y','z','v']].values
                
                # Allow new points to join (relaxed max tracked points)
                if frame_idx == 0:
                    for i in range(min(len(current_points), self.max_points)):
                        tracks[i] = [current_points[i]]
                else:
                    # Simple nearest neighbor matching (instead of Hungarian algorithm)
                    if tracks:
                        prev_points = np.array([t[-1] for t in tracks.values()])
                        distances = cdist(prev_points[:,:3], current_points[:,:3])
                        matches = np.argmin(distances, axis=1)
                        
                        # Update existing tracks
                        for track_id, match_idx in enumerate(matches):
                            if track_id < len(tracks):
                                tracks[track_id].append(current_points[match_idx])
                                
                    # Add new tracks (keep empty slots)
                    for i in range(len(current_points)):
                        if i not in tracks and len(tracks) < self.max_points:
                            tracks[len(tracks)] = [current_points[i]]
            
            # Pad to uniform size (allow zero padding)
            padded_sequence = np.zeros((self.window_size, self.max_points, 4))
            for track_id in tracks:
                if track_id >= self.max_points:
                    continue
                track_length = len(tracks[track_id])
                padded_sequence[:track_length, track_id] = tracks[track_id]
                
            return padded_sequence
        except:
            return None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        
        # Randomly sample time steps
        time_steps = np.random.choice(
            sequence.shape[0], 
            self.window_size, 
            replace=sequence.shape[0] < self.window_size
        )
        
        # Normalization
        normalized = sequence[time_steps]
        normalized[..., :3] = (normalized[..., :3] - np.mean(normalized[..., :3])) / (np.std(normalized[..., :3]) + 1e-8)
        normalized[..., 3] = (normalized[..., 3] - np.mean(normalized[..., 3])) / (np.std(normalized[..., 3]) + 1e-8)
        
        return torch.FloatTensor(normalized), label
# ----------------------
# Model definition module
# ----------------------
class TNet(nn.Module):
    """Spatial Transformer Network"""
    def __init__(self, k=6):
        super().__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc = nn.Linear(1024, k*k)
        self.k = k
        
    def forward(self, x):
        x = x.transpose(2, 1)
        bs = x.size(0)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max(self.conv3(x), 2, keepdim=True)[0]
        x = self.fc(x.view(bs, -1))
        iden = torch.eye(self.k, device=x.device).view(1, self.k*self.k).repeat(bs, 1)
        return x.view(bs, self.k, self.k) + iden.view(bs, self.k, self.k)

class RadarPointNet(nn.Module):
    """Enhanced PointNet (adapted for 4 input features)"""
    def __init__(self):
        super().__init__()
        self.input_tnet = TNet(k=4)  # Changed to 4-dimensional input
        self.mlp = nn.Sequential(
            nn.Conv1d(4, 64, 1),    # Input channels changed to 4
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [B, T, N, 4]
        B, T, N, _ = x.shape
        x = x.view(B*T, N, 4)  # Adjusted to 4 features
        
        # Apply spatial transformation (consistent with input dimension)
        trans = self.input_tnet(x)
        x = torch.bmm(x, trans)
        
        # Feature extraction (input dimension matched)
        x = x.transpose(2, 1)  # [B*T, 4, N]
        x = self.mlp(x)        # [B*T, 256, N]
        global_feat = torch.max(x, 2)[0]
        return global_feat.view(B, T, -1)

class RadarActivityClassifier(nn.Module):
    """Spatiotemporal Joint Classifier"""
    def __init__(self, num_classes):
        super().__init__()
        self.pointnet = RadarPointNet()
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x: [B, T, N, 6]
        B, T, N, _ = x.size()
        point_feats = self.pointnet(x)  # [B, T, 256]
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(point_feats)  # [B, T, 256]
        
        # Temporal average pooling
        temporal_feat = torch.mean(lstm_out, dim=1)  # [B, 256]
        return self.classifier(temporal_feat)

# ----------------------
# Training and evaluation module
# ----------------------
def train_model(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
    
    for seqs, labels in progress_bar:
        seqs = seqs.to(config["device"])
        labels = labels.to(config["device"])
        
        optimizer.zero_grad()
        outputs = model(seqs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(train_loader)
    logging.info(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")
    return avg_loss

# ----------------------
# Enhanced evaluation module
# ----------------------
def safe_evaluate(model, test_loader, num_classes):
    """Safe evaluation method to prevent metric failure"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    class_counts = np.zeros(num_classes)
    
    with torch.no_grad():
        for seqs, labels in test_loader:
            seqs = seqs.to(config["device"])
            outputs = model(seqs)
            
            # Record class distribution
            unique, counts = np.unique(labels.numpy(), return_counts=True)
            for u, c in zip(unique, counts):
                class_counts[u] += c
                
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.append(probs.cpu().numpy())
    
    # Check class validity
    valid_classes = np.where(class_counts > 0)[0]
    print(f"Valid classes: {len(valid_classes)}/{num_classes}")
    print("Class distribution:", class_counts)
    
    # Regenerate valid label mapping
    label_map = {old: new for new, old in enumerate(valid_classes)}
    inverse_map = {v: k for k, v in label_map.items()}
    
    # Filter out invalid classes
    filtered_labels = [label_map[l] for l in all_labels if l in label_map]
    filtered_preds = [label_map[p] for p, l in zip(all_preds, all_labels) if l in label_map]
    filtered_probs = np.array([
        p[list(valid_classes)] 
        for p, l in zip(np.concatenate(all_probs, axis=0), all_labels)
        if l in valid_classes
    ])
    
    # Calculate safe metrics
    metrics = {}
    if len(valid_classes) > 1:
        try:
            metrics['auc'] = roc_auc_score(
                label_binarize(filtered_labels, classes=np.arange(len(valid_classes))),
                filtered_probs,
                multi_class='ovr'
            )
        except Exception as e:
            metrics['auc'] = np.nan
    else:
        metrics['auc'] = np.nan
    
    # Calculate basic metrics
    metrics.update({
        'accuracy': accuracy_score(filtered_labels, filtered_preds),
        'f1': f1_score(filtered_labels, filtered_preds, average='weighted', zero_division=0),
        'recall': recall_score(filtered_labels, filtered_preds, average='weighted', zero_division=0),
        'precision': precision_score(filtered_labels, filtered_preds, average='weighted', zero_division=0)
    })
    
    return metrics, inverse_map

# ----------------------
# Main program
# ----------------------
if __name__ == "__main__":
    # Initialize label processor with multiple data roots
    # Use data_roots if available, otherwise fall back to single data_root
    data_sources = config.get("data_roots", [config["data_root"]])
    print(f"\nLoading data from {len(data_sources)} source(s):")
    for i, src in enumerate(data_sources, 1):
        print(f"  {i}. {src}")
    
    label_processor = LabelProcessor(data_sources)
    
    # Create dataset
    full_dataset = RadarSequenceDataset(
        label_processor, 
        window_size=config["window_size"]
    )
    
    # Split train and test sets
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    
    # Initialize model
    model = RadarActivityClassifier(label_processor.num_classes).to(config["device"])
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_auc = 0
    for epoch in range(config["num_epochs"]):
        train_loss = train_model(model, train_loader, criterion, optimizer, epoch)
        test_metrics, label_map = safe_evaluate(model, test_loader, label_processor.num_classes)
        
        # # Print interpretable results
        # print("\nClass mapping table:")
        # for new_id, old_id in label_map.items():
        #     print(f"{new_id} -> {old_id}: {label_processor.valid_labels[old_id]['name']}")
            
        # Record evaluation results
        log_msg = (
            f"Epoch {epoch+1} Test Results:\n"
            f"  Accuracy: {test_metrics['accuracy']:.4f}\n"
            f"  F1-Score: {test_metrics['f1']:.4f}\n"
            f"  Recall: {test_metrics['recall']:.4f}\n"
            f"  Precision: {test_metrics['precision']:.4f}\n"
            f"  AUC-ROC: {test_metrics['auc']:.4f}"
        )
        logging.info(log_msg)
        print(log_msg)
        
        # # Save best model
        # if test_metrics['auc'] > best_auc:
        #     best_auc = test_metrics['auc']
        #     torch.save(model.state_dict(), os.path.join(config["log_dir"], "best_model.pth"))
        #     logging.info(f"New best model saved with AUC: {best_auc:.4f}")