import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from data_utils.imu_dataloader import IMU_DataLoader
from models.cnn import SupervisedCNN1D, kaiming_init_weights
from models.cnn_transformer import SupervisedTransformer  # Add this import
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import json
import argparse

def calculate_metrics(y_true, y_pred, y_pred_proba):
    # Convert numpy arrays to lists for JSON serialization
    metrics = {
        'accuracy': float((y_true == y_pred).mean() * 100),
        'f1': float(f1_score(y_true, y_pred, average='macro') * 100),
        'precision': float(precision_score(y_true, y_pred, average='macro') * 100),
        'recall': float(recall_score(y_true, y_pred, average='macro') * 100),
    }
    
    # For ROC AUC, we need to binarize the labels
    try:
        metrics['roc_auc'] = float(roc_auc_score(
            y_true=pd.get_dummies(y_true), 
            y_score=y_pred_proba,
            multi_class='ovr',
            average='macro'
        ) * 100)
    except:
        metrics['roc_auc'] = float('nan')  # In case of errors (e.g., single class)
        
    return metrics

def train_epoch(model, train_loader, criterion, optimizer, device, modalities):
    model.train()
    total_loss = 0
    all_labels = []
    all_predictions = []
    all_predictions_proba = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        x = batch[modalities].to(device)
        y = batch['label'].to(device)
        
        outputs = model(x)
        loss = criterion(outputs, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Store predictions and labels
        _, predicted = outputs.max(1)
        all_labels.extend(y.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_predictions_proba.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())
        
        # Update progress bar with loss only
        pbar.set_postfix({'loss': total_loss/len(train_loader)})
    
    # Convert numpy arrays to lists for JSON serialization
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_predictions_proba)
    )
    metrics['loss'] = float(total_loss/len(train_loader))
    
    return metrics, [int(x) for x in all_labels], [int(x) for x in all_predictions]

def evaluate(model, test_loader, criterion, device, modalities):
    model.eval()
    total_loss = 0
    all_labels = []
    all_predictions = []
    all_predictions_proba = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            x = batch[modalities].to(device)
            y = batch['label'].to(device)
            
            outputs = model(x)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            
            # Store predictions and labels
            _, predicted = outputs.max(1)
            all_labels.extend(y.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_predictions_proba.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())
    
    # Convert numpy arrays to lists for JSON serialization
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_predictions_proba)
    )
    metrics['loss'] = float(total_loss/len(test_loader))
    
    return metrics, [int(x) for x in all_labels], [int(x) for x in all_predictions]

def main(data_path, modalities, sample_length, batch_size, num_epochs, learning_rate, device, in_channels, out_size, random_init=False, experiment_name=None, args=None):
    
    # Create data loaders
    train_dataset = IMU_DataLoader(
        datapth=data_path,
        data_type='train',  # Changed from 'train' to 'supervised'
        modalities=modalities,
        sample_length=sample_length
    )
    
    test_dataset = IMU_DataLoader(
        datapth=data_path,
        data_type='test',
        modalities=modalities,
        sample_length=sample_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    # import pdb; pdb.set_trace()
    
    # Create model
    model = SupervisedTransformer(
    in_channels=in_channels,
    sample_length=sample_length,
    out_size=out_size,
    modality=modalities,  # Use the modalities parameter
    device=device
)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Create directories for logging
    # experiment_name = data_path.split('/')[-2]
    
    run_name=data_path.split('/')[-1].split('.')[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('runs', experiment_name, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save args separately as well for easy access
    if args:
        args_file = os.path.join(run_dir, 'args.json')
        with open(args_file, 'w') as f:
            json.dump(vars(args), f, indent=4)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(run_dir)
    
    # Initialize log file for epoch outputs
    log_file = os.path.join(run_dir, 'training_log.txt')
    
    def log_message(message):
        """Write message to both console and log file"""
        print(message)
        with open(log_file, 'a') as f:
            f.write(message + '\n')
    
    # Write header to log file
    with open(log_file, 'w') as f:
        f.write(f"Training Log - {timestamp}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Data Path: {data_path}\n")
        f.write(f"Modalities: {modalities}\n")
        f.write(f"Sample Length: {sample_length}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Num Epochs: {num_epochs}\n")
        f.write(f"Device: {device}\n")
        f.write(f"{'='*60}\n\n")
    
    # Initialize best results dictionary
    best_results = {
        'timestamp': timestamp,
        'best_metrics': None,
        'best_epoch': None,
        'best_predictions': None,
        'best_labels': None,
        'args': vars(args) if args else None,  # Save all command-line arguments
        'training_config': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'sample_length': sample_length,
            'device': str(device),
            'in_channels': in_channels,
            'out_size': out_size,
            'data_path': data_path,
            'modalities': modalities,
            'random_init': random_init,
            'experiment_name': experiment_name
        }
    }
    
    if random_init:
        # Only evaluate with random initialization
        print('\nEvaluating with random initialization')
        # model.apply(kaiming_init_weights)
        # Freeze conv_layers
        for param in model.conv_layers.parameters():
            param.requires_grad = False
        
        
        # test_metrics, test_labels, test_preds = evaluate(
        #     model, test_loader, criterion, device, modalities
        # )
        # print('Testing metrics:')
        # for metric, value in test_metrics.items():
        #     print(f'{metric}: {value:.2f}')
        
        # # Save results
        # best_results['best_metrics'] = test_metrics
        # best_results['best_epoch'] = 0
        # best_results['best_predictions'] = test_preds
        # best_results['best_labels'] = test_labels
        
        # # Save results to JSON
        # results_file = os.path.join(run_dir, 'random_init_results.json')
        # with open(results_file, 'w') as f:
        #     json.dump(best_results, f, indent=4)
            
        # writer.close()
        # return
    
    # Training loop
    # model.apply(kaiming_init_weights) # update at 0616
    best_acc = 0
    for epoch in range(num_epochs):
        log_message(f'\nEpoch {epoch+1}/{num_epochs}')
        
        
        
        # Train
        train_metrics, train_labels, train_preds = train_epoch(
            model, train_loader, criterion, optimizer, device, modalities
        )
        log_message('Training metrics:')
        for metric, value in train_metrics.items():
            log_message(f'{metric}: {value:.2f}')
            writer.add_scalar(f'train/{metric}', value, epoch)
        
        # Evaluate
        test_metrics, test_labels, test_preds = evaluate(
            model, test_loader, criterion, device, modalities
        )
        log_message('Testing metrics:')
        for metric, value in test_metrics.items():
            log_message(f'{metric}: {value:.2f}')
            writer.add_scalar(f'test/{metric}', value, epoch)
        
        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, epoch)
        log_message(f'Learning rate: {current_lr:.6f}')
        scheduler.step(test_metrics['loss'])
        
        # Save best model and results
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            
            log_message(f'New best accuracy: {best_acc:.2f}%')
            
            # Update best results
            best_results['best_metrics'] = {
                'accuracy': float(test_metrics['accuracy']),
                'f1': float(test_metrics['f1']),
                'precision': float(test_metrics['precision']),
                'recall': float(test_metrics['recall']),
                'roc_auc': float(test_metrics['roc_auc']),
                'loss': float(test_metrics['loss'])
            }
            best_results['best_epoch'] = epoch + 1
            best_results['best_predictions'] = [int(x) for x in test_preds]
            best_results['best_labels'] = [int(x) for x in test_labels]
            
            # Log best metrics to TensorBoard
            for metric, value in best_results['best_metrics'].items():
                writer.add_scalar(f'best/{metric}', value, epoch)
    
    # Write final summary to log
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Training Completed\n")
        f.write(f"Best Epoch: {best_results['best_epoch']}\n")
        f.write(f"Best Accuracy: {best_acc:.2f}%\n")
        if best_results['best_metrics']:
            f.write(f"Best Metrics:\n")
            for metric, value in best_results['best_metrics'].items():
                f.write(f"  {metric}: {value:.2f}\n")
        f.write(f"{'='*60}\n")
    
    # Save best results to JSON
    results_file = os.path.join(run_dir, 'best_results.json')
    with open(results_file, 'w') as f:
        json.dump(best_results, f, indent=4)
    
    writer.close()

    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data_imu/activity_40/dataset_fold_1.csv')
    parser.add_argument('--modalities', type=str, default='acc')
    parser.add_argument('--sample_length', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--random_init', action='store_true', help='Only evaluate with random initialization')
    parser.add_argument('--experiment_name', type=str, default='activity_40')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    
    # Create necessary directories
    os.makedirs('runs', exist_ok=True)
    
    # Data parameters
    data_path = args.data_path
    modalities = args.modalities
    sample_length = args.sample_length
    batch_size = args.batch_size
    
    # Training parameters
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    device = args.device
    random_init = args.random_init
    experiment_name=args.experiment_name
    
    # Model parameters
    if modalities in ['acc','gyr','mag']:
        in_channels = 15  # 5 sensors * 3 axes
    elif modalities in ['acc_gyr','acc_mag','gyr_mag']:
        in_channels = 30  # 5 sensors * 3 axes * 2 modalities
    elif modalities=='acc_gyr_mag':
        in_channels = 45  # 5 sensors * 3 axes * 3 modalities
    out_size = pd.read_csv(data_path)['activity_id_num'].nunique()  # number of activity classes
    main(data_path, modalities, sample_length, batch_size, num_epochs, learning_rate, device, in_channels, out_size, random_init, experiment_name, args)