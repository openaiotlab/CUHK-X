import os
import csv
import pandas as pd
import numpy as np
import re
from bert_score import score
from rouge import Rouge  
import argparse
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def read_csv_with_activities(csv_path):
    """
    读取CSV文件，提取path, logic和vlm_result列
    """
    data = []
    model_name = os.path.basename(csv_path).split('_')[1].split('.')[0]
    print(f"检测到模型名称: {model_name}")
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # 跳过表头
            
            # 找到正确的列索引
            path_idx = header.index("Path") if "Path" in header else 0
            logic_idx = header.index("Logic") if "Logic" in header else 1
            result_idx = header.index("vlm_result") if "vlm_result" in header else 2
            
            for row in reader:
                if len(row) > max(path_idx, logic_idx, result_idx):
                    path = row[path_idx]
                    logic = row[logic_idx]
                    vlm_result = row[result_idx]
                    
                    # 特殊处理videochatr1模型的输出格式 - 仅用于日志记录
                    if model_name == "videochatr1" and isinstance(vlm_result, str) and "<answer>" in vlm_result:
                        print(f"检测到videochatr1格式的回答")
                    
                    data.append({
                        "path": path,
                        "logic": logic,
                        "vlm_result": vlm_result,
                        "model_name": model_name
                    })
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        import traceback
        traceback.print_exc()
    
    return data

def normalize_activities(activity_str):
    """
    标准化活动字符串：去除多余空格，转为小写，拆分为列表
    同时处理可能包含<answer>标签的情况
    """
    if not activity_str:
        return []
    
    if not isinstance(activity_str, str):
        activity_str = str(activity_str)
    
    # 处理可能包含的<answer>标签
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.search(answer_pattern, activity_str, re.DOTALL)
    if match:
        # 提取<answer>和</answer>之间的内容
        activity_str = match.group(1).strip()
    
    # 将字符串分割成单个活动，支持英文逗号和中文逗号
    activity_str = activity_str.replace('，', ',')
    activities = [act.strip().lower() for act in activity_str.split(',')]
    # 过滤掉空字符串
    activities = [act for act in activities if act]
    return activities

def calculate_activity_precision_recall(gt_activities, pred_activities):
    """计算活动预测的精确率和召回率"""
    gt_set = set(gt_activities)
    pred_set = set(pred_activities)
    
    if not gt_set:
        return 0.0, 0.0, 0.0
    
    if not pred_set:
        return 0.0, 0.0, 0.0
        
    common = gt_set.intersection(pred_set)
    precision = len(common) / len(pred_set) if pred_set else 0.0
    recall = len(common) / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def calculate_missing_activities(gt_activities, pred_activities):
    """计算未被预测出的活动"""
    gt_set = set(gt_activities)
    pred_set = set(pred_activities)
    missing = gt_set - pred_set
    return list(missing)

def calculate_extra_activities(gt_activities, pred_activities):
    """计算预测出但不在GT中的活动"""
    gt_set = set(gt_activities)
    pred_set = set(pred_activities)
    extra = pred_set - gt_set
    return list(extra)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='internvl2B', help='Model name')
    parser.add_argument('--modality', type=str, default='rgb', help='depth, rgb, ir')

    args = parser.parse_args()

    method = args.method 
    modality = args.modality  # 'depth', 'rgb', 'ir' 
   
    # 读取预测结果CSV文件
    pred_path = f'CUHK-X-VLM/src/action_selection/predictions/{modality}/pred_{method}.csv'
    data = read_csv_with_activities(pred_path)
    print(f"Loaded {len(data)} samples from {pred_path}")
    print(f"使用模型: {method}")

    # 初始化评分统计
    total_samples = 0
    valid_samples = 0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_complete_predictions = 0  # 完全预测出所有活动的样本数
    
    # 所有GT活动的统计
    all_gt_activities = set()
    # 未被预测出的活动统计
    missing_activity_counts = {}
    
    # 逐行比较活动预测
    for item in data:
        path = item["path"]
        gt = item["logic"]
        vlm_result = item["vlm_result"]
        model_name = item.get("model_name", method)
        
        # 检查是否有无效结果
        if vlm_result in ["OOM", "video cannot find"] or (isinstance(vlm_result, str) and vlm_result.startswith("ERROR:")):
            print(f"跳过无效样本: {path}")
            continue
            
        # 标准化活动列表，自动处理<answer>标签
        gt_activities = normalize_activities(gt)
        pred_activities = normalize_activities(vlm_result)
        
        # 打印原始和处理后的预测结果，用于调试
        if isinstance(vlm_result, str) and "<answer>" in vlm_result:
            print(f"原始预测: {vlm_result}")
            print(f"处理后预测: {pred_activities}")
        
        # 更新所有GT活动的统计
        all_gt_activities.update(gt_activities)
        
        # 计算精确率和召回率
        precision, recall, f1 = calculate_activity_precision_recall(gt_activities, pred_activities)
        
        # 计算未预测出的活动
        missing_activities = calculate_missing_activities(gt_activities, pred_activities)
        for act in missing_activities:
            missing_activity_counts[act] = missing_activity_counts.get(act, 0) + 1
        
        # 计算额外预测的活动
        extra_activities = calculate_extra_activities(gt_activities, pred_activities)
        
        is_complete = len(missing_activities) == 0
        
        print(f"\n样本: {path}")
        print(f"GT活动: {gt_activities}")
        print(f"预测活动: {pred_activities}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"是否完全预测: {'是' if is_complete else '否'}")
        if missing_activities:
            print(f"未预测出的活动: {missing_activities}")
        if extra_activities:
            print(f"额外预测的活动: {extra_activities}")
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        if is_complete:
            total_complete_predictions += 1
        valid_samples += 1
        total_samples += 1
    
    # 计算平均分数
    if valid_samples > 0:
        avg_precision = total_precision / valid_samples
        avg_recall = total_recall / valid_samples
        avg_f1 = total_f1 / valid_samples
        complete_ratio = total_complete_predictions / valid_samples
        
        print("\n=== 总体评估结果 ===")
        print(f"总样本数: {total_samples}")
        print(f"有效样本数: {valid_samples}")
        print(f"平均精确率: {avg_precision:.4f}")
        print(f"平均召回率: {avg_recall:.4f}")
        print(f"平均F1分数: {avg_f1:.4f}")
        print(f"完全预测比例: {complete_ratio:.4f} ({total_complete_predictions}/{valid_samples})")
        
        # 输出所有GT活动
        print(f"\n总共有 {len(all_gt_activities)} 种不同的活动")
        
        # 输出未被预测出的活动统计
        if missing_activity_counts:
            print("\n未被预测出的活动统计:")
            sorted_missing = sorted(missing_activity_counts.items(), key=lambda x: x[1], reverse=True)
            for act, count in sorted_missing:
                miss_rate = count / valid_samples
                print(f"  {act}: {count}次 (占样本比例: {miss_rate:.2%})")
    else:
        print("没有有效样本进行评估")
        sorted_missing = []
    
    # 保存评估结果
    results_dir = f'CUHK-X-VLM/src/action_selection/scores/{modality}'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    
    output_csv = f'{results_dir}/{method}_activity_scores.csv'
    with open(output_csv, mode="w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total_Samples", total_samples])
        writer.writerow(["Valid_Samples", valid_samples])
        writer.writerow(["Avg_Precision", avg_precision if valid_samples > 0 else 0])
        writer.writerow(["Avg_Recall", avg_recall if valid_samples > 0 else 0])
        writer.writerow(["Avg_F1", avg_f1 if valid_samples > 0 else 0])
        writer.writerow(["Complete_Prediction_Ratio", complete_ratio if valid_samples > 0 else 0])
    
    # 保存活动统计结果
    if valid_samples > 0:
        activity_stats_csv = f'{results_dir}/{method}_activity_stats.csv'
        with open(activity_stats_csv, mode="w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Activity", "Missing_Count", "Missing_Rate"])
            for act, count in sorted_missing:
                miss_rate = count / valid_samples
                writer.writerow([act, count, f"{miss_rate:.4f}"])
        
        print(f"活动统计结果已保存到 {activity_stats_csv}")
    
    print(f"\n评估结果已保存到 {output_csv}")