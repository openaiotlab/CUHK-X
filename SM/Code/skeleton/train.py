'''Example training commands:

#cross_trial/intra dataset
CUDA_VISIBLE_DEVICES=4,6 python train.py --train_dir cross_trial_train.txt --test_dir cross_trial_test.txt --config ./configs/dstformer.yaml

#cross_subject dataset with all 30 users
CUDA_VISIBLE_DEVICES=1,3 python train.py --train_dir cross_subject_train_30.txt --test_dir cross_subject_test_30.txt --config ./configs/dstformer.yaml

#resampled dataset with 20 users
CUDA_VISIBLE_DEVICES=1,3 python train.py --train_dir cross_subject_train_top20_test1.txt --test_dir cross_subject_test_top20_test1.txt --config ./configs/dstformer.yaml

CUDA_VISIBLE_DEVICES=3 python train.py --train_dir cross_subject_train_10 --test_dir cross_subject_test_10 --config configs/dstformer.yaml --evaluate checkpoint/cuhkx_cross_subject/DSTformer_cross_subject_train_10_resmapling_smallermodel/best_epoch.bin

'''
import os
import numpy as np
import time
import sys
import argparse
import errno
from collections import OrderedDict
import tensorboardX
from tqdm import tqdm
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from utils.tools import *
from utils.learning import *
from utils.dataset import cuhkxDataset, create_weighted_sampler
from model.loss import *
from model.model_action import ActionNet

import wandb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report
    
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-freq', '--print_freq', default=100)
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('--train_dir', default='train', type=str, metavar='PATH', help='train files')
    parser.add_argument('--test_dir', default='test', type=str, metavar='PATH', help='test files')

    opts = parser.parse_args()
    return opts

def validate(test_loader, model, criterion):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    all_targets = []
    all_preds = []
    all_probs = []
    minority_classes = [33, 40, 35, 42, 25]
    majority_classes = [36, 23, 9, 17, 12]
    with torch.no_grad():
        end = time.time()
        for idx, (batch_input, batch_gt) in tqdm(enumerate(test_loader)):
            batch_size = len(batch_input)    
            if torch.cuda.is_available():
                batch_gt = batch_gt.cuda()
                batch_input = batch_input.cuda()
            output = model(batch_input)    # (N, num_classes)
            
            loss = criterion(output, batch_gt)

            # update metric
            losses.update(loss.item(), batch_size)
            acc1, acc5 = accuracy(output, batch_gt, topk=(1, 5))
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # collect for sklearn metrics
            preds = torch.argmax(output, dim=1).detach().cpu().numpy()
            targets = batch_gt.detach().cpu().numpy()
            probs = torch.softmax(output, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)
            all_probs.extend(probs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    # Compute sklearn metrics
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    y_true = np.array(all_targets)
    y_score = np.array(all_probs)

    unique_classes = np.unique(y_true)
    y_score_sub = y_score[:, unique_classes]

    # renormalize y_score_sub to ensure it sums to 1 for each sample
    y_score_sub = y_score_sub / y_score_sub.sum(axis=1, keepdims=True)

    try:
        auc_roc = roc_auc_score(
            y_true,
            y_score_sub,
            labels=unique_classes,
            average='macro',
            multi_class='ovr'
        )
    except Exception as e:
        print("roc_auc_score error:", e)
        auc_roc = float('nan')
    
    
    minority_metrics = {}
    class_report = classification_report(y_true, all_preds, labels=unique_classes, 
                                       target_names=[str(cls) for cls in unique_classes], 
                                       output_dict=True, zero_division=0)
    print(class_report)
    print("\n=== Minority Classes Performance ===")
    for cls in minority_classes:
        if cls in unique_classes:
            cls_str = str(cls)
            if cls_str in class_report:
                cls_metrics = class_report[cls_str]
                cls_count = (y_true == cls).sum()
                
                minority_metrics[cls] = {
                    'precision': cls_metrics['precision'],
                    'recall': cls_metrics['recall'], 
                    'f1': cls_metrics['f1-score'],
                    'accuracy': cls_metrics['recall'],  # 对单个类别，recall就是accuracy
                    'sample_count': cls_count
                }
                
                print(f"Class {cls}: Acc={cls_metrics['recall']:.4f}, "
                      f"Precision={cls_metrics['precision']:.4f}, "
                      f"Recall={cls_metrics['recall']:.4f}, "
                      f"F1={cls_metrics['f1-score']:.4f}, "
                      f"Samples={cls_count}")
            else:
                print(f"Class {cls}: No samples found in test set")
                minority_metrics[cls] = None
        else:
            print(f"Class {cls}: Not present in test set")
            minority_metrics[cls] = None

    # 计算存在的少数类的平均性能
    valid_metrics = [m for m in minority_metrics.values() if m is not None]
    if valid_metrics:
        avg_minority_precision = np.mean([m['precision'] for m in valid_metrics])
        avg_minority_recall = np.mean([m['recall'] for m in valid_metrics])
        avg_minority_f1 = np.mean([m['f1'] for m in valid_metrics])
        avg_minority_acc = np.mean([m['accuracy'] for m in valid_metrics])
        
        print(f"\nAverage Minority Classes Performance:")
        print(f"Avg Accuracy: {avg_minority_acc:.4f}")
        print(f"Avg Precision: {avg_minority_precision:.4f}")
        print(f"Avg Recall: {avg_minority_recall:.4f}")
        print(f"Avg F1: {avg_minority_f1:.4f}")
    else:
        print("No minority classes found in test set")
        avg_minority_precision = avg_minority_recall = avg_minority_f1 = avg_minority_acc = 0.0

    # Majority classes performance
    print("\n=== Majority Classes Performance ===")
    majority_metrics = {}
    for cls in majority_classes:
        if cls in unique_classes:
            cls_str = str(cls)
            if cls_str in class_report:
                cls_metrics = class_report[cls_str]
                cls_count = (y_true == cls).sum()
                
                majority_metrics[cls] = {
                    'precision': cls_metrics['precision'],
                    'recall': cls_metrics['recall'], 
                    'f1': cls_metrics['f1-score'],
                    'accuracy': cls_metrics['recall'],  # 对单个类别，recall就是accuracy
                    'sample_count': cls_count
                }
                
                print(f"Class {cls}: Acc={cls_metrics['recall']:.4f}, "
                      f"Precision={cls_metrics['precision']:.4f}, "
                      f"Recall={cls_metrics['recall']:.4f}, "
                      f"F1={cls_metrics['f1-score']:.4f}, "
                      f"Samples={cls_count}")
            else:
                print(f"Class {cls}: No samples found in test set")
                majority_metrics[cls] = None
        else:
            print(f"Class {cls}: Not present in test set")
            majority_metrics[cls] = None
    # 计算存在的多数类的平均性能
    valid_majority_metrics = [m for m in majority_metrics.values() if m is not None]
    if valid_majority_metrics:
        avg_majority_precision = np.mean([m['precision'] for m in valid_majority_metrics])
        avg_majority_recall = np.mean([m['recall'] for m in valid_majority_metrics])
        avg_majority_f1 = np.mean([m['f1'] for m in valid_majority_metrics])
        avg_majority_acc = np.mean([m['accuracy'] for m in valid_majority_metrics])
        
        print(f"\nAverage Majority Classes Performance:")
        print(f"Avg Accuracy: {avg_majority_acc:.4f}")
        print(f"Avg Precision: {avg_majority_precision:.4f}")
        print(f"Avg Recall: {avg_majority_recall:.4f}")
        print(f"Avg F1: {avg_majority_f1:.4f}")

    print('Test Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Acc@1 {top1.val:.4f} ({top1.avg:.4f})\t'
            'Acc@5 {top5.val:.4f} ({top5.avg:.4f})\t'
            'Precision {precision:.4f}\t'
            'Recall {recall:.4f}\t'
            'F1 {f1:.4f}\t'
            'AUC-ROC {auc_roc:.4f}'.format( batch_time=batch_time,
            loss=losses, top1=top1, top5=top5,
            precision=precision, recall=recall, f1=f1, auc_roc=auc_roc))
    return losses.avg, top1.avg, top5.avg, precision, recall, f1, auc_roc


def train_with_config(args, opts):
    print(args)
    model_backbone = load_backbone(args)
    model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes, dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim, num_joints=args.num_joints)

    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda() 
    best_acc = 0
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)
    print('Loading dataset...')
    train_dataset = cuhkxDataset(data_split='split_data_results/'+opts.train_dir, scale_range=args.scale_range_train)
    val_dataset = cuhkxDataset(data_split='split_data_results/'+opts.test_dir, scale_range=args.scale_range_test)
    
    labels = train_dataset.labels
    sampler = create_weighted_sampler(labels)
    trainloader_params = {
          'batch_size': args.batch_size,
        #   'shuffle': True,
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True,
          'drop_last': True,
          'sampler': sampler    
    }
    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    train_loader = DataLoader(train_dataset, **trainloader_params)
    test_loader = DataLoader(val_dataset, **testloader_params)

    if not opts.evaluate:
        opts.checkpoint = 'checkpoint/cuhkx_cross_trial/'+ args.backbone + '_' + opts.train_dir+'_resmapling'+'_smallermodel_2nd_try'
        os.makedirs(opts.checkpoint, exist_ok=True)
        chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
        if os.path.exists(chk_filename):
            opts.resume = chk_filename
    if opts.resume or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.resume
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        
        # Check if the checkpoint has the correct number of classes
        checkpoint_num_classes = checkpoint['model']['module.head.fc2.weight'].shape[0]
        current_num_classes = args.action_classes
        
        if checkpoint_num_classes != current_num_classes:
            print(f'WARNING: Checkpoint has {checkpoint_num_classes} classes, but current model has {current_num_classes} classes.')
            print('Skipping checkpoint loading and training from scratch.')
            if opts.evaluate:
                raise ValueError(f'Cannot evaluate: checkpoint classes ({checkpoint_num_classes}) != model classes ({current_num_classes})')
            opts.resume = ''  # Clear resume flag to train from scratch
        else:
            model.load_state_dict(checkpoint['model'], strict=True)
    
    if not opts.evaluate:
        optimizer = optim.AdamW(
            [     {"params": filter(lambda p: p.requires_grad, model.module.backbone.parameters()), "lr": args.lr_backbone},
                  {"params": filter(lambda p: p.requires_grad, model.module.head.parameters()), "lr": args.lr_head},
            ],      lr=args.lr_backbone, 
                    weight_decay=args.weight_decay
        )

        # scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
        st = 0
        print('INFO: Training on {} batches'.format(len(train_loader)))
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            lr = checkpoint['lr']
            if 'best_acc' in checkpoint and checkpoint['best_acc'] is not None:
                best_acc = checkpoint['best_acc']
        # Training
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            losses_train = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            model.train()
            end = time.time()
            iters = len(train_loader)
            for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):    # (N, 2, T, 17, 3)
                data_time.update(time.time() - end)
                batch_size = len(batch_input)
                if torch.cuda.is_available():
                    batch_gt = batch_gt.cuda()
                    batch_input = batch_input.cuda()
                output = model(batch_input) # (N, num_classes)
                optimizer.zero_grad()
                loss_train = criterion(output, batch_gt)
                losses_train.update(loss_train.item(), batch_size)
                acc1, acc5 = accuracy(output, batch_gt, topk=(1, 5))
                top1.update(acc1[0], batch_size)
                top5.update(acc5[0], batch_size)
                loss_train.backward()
                optimizer.step()    
                batch_time.update(time.time() - end)
                end = time.time()
            if (idx + 1) % opts.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       epoch, idx + 1, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses_train, top1=top1))
                sys.stdout.flush()

            test_loss, test_top1, test_top5, test_precision, test_recall, test_f1, test_auc_roc = validate(test_loader, model, criterion)

            wandb.log({
                'train_loss': losses_train.avg,
                'train_top1': top1.avg,
                'train_top5': top5.avg,
                'test_loss': test_loss,
                'test_top1': test_top1,
                'test_top5': test_top5,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'test_auc_roc': test_auc_roc,
            }, step=epoch + 1)

            scheduler.step()

            # Save latest checkpoint.
            chk_path = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            print('Saving checkpoint to', chk_path)
            torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc' : best_acc
            }, chk_path)

            # Save best checkpoint.
            best_chk_path = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))
            if test_top1 > best_acc:
                best_acc = test_top1
                print("save best checkpoint")
                torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc' : best_acc
                }, best_chk_path)

    if opts.evaluate:
        test_loss, test_top1, test_top5, precision, recall, f1, auc_roc = validate(test_loader, model, criterion)
        print('Loss {loss:.4f} \t'
              'Acc@1 {top1:.4f} \t'
              'Precision {precision:.4f} \t'
              'Recall {recall:.4f} \t'
              'F1 {f1:.4f} \t'
              'AUC-ROC {auc_roc:.4f} \t'.format(loss=test_loss, top1=test_top1, precision=precision, recall=recall, f1=f1, auc_roc=auc_roc))

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    if not opts.evaluate:
        wandb.init(project="cuhkx_cross_trial", name=args.backbone+'_'+opts.train_dir+'_resmapling'+'_smallermodel_2nd_try')
    train_with_config(args, opts)
    