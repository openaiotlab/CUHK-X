import os
import csv
import pandas as pd
import numpy as np
from bert_score import score
from rouge import Rouge  
import argparse
import nltk
import re
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
            logic_idx = header.index("Original_GT") if "Original_GT" in header else 1
            shuffled_idx = header.index("Shuffled_Activities") if "Shuffled_Activities" in header else 2
            result_idx = header.index("vlm_result") if "vlm_result" in header else 3
            
            for row in reader:
                if len(row) > max(path_idx, logic_idx, result_idx):
                    path = row[path_idx]
                    logic = row[logic_idx]
                    shuffled = row[shuffled_idx] if len(row) > shuffled_idx else ""
                    vlm_result = row[result_idx]
                    
                    # 特殊处理videochatr1模型的输出格式 - 仅用于日志记录
                    if model_name == "videochatr1" and isinstance(vlm_result, str) and "<answer>" in vlm_result:
                        print(f"检测到videochatr1格式的回答")
                        
                    data.append({
                        "path": path,
                        "logic": logic,
                        "shuffled": shuffled,
                        "vlm_result": vlm_result
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
    
    # 替换中文逗号为英文逗号
    activity_str = activity_str.replace('，', ',')
    
    # 将字符串分割成单个活动
    activities = [act.strip().lower() for act in activity_str.split(',')]
    # 过滤掉空字符串
    activities = [act for act in activities if act]
    return activities

def calculate_order_score(gt_activities, pred_activities):
    """
    计算两个活动序列之间的顺序一致性评分
    
    算法：
    1. 只考虑两个序列中共同存在的活动
    2. 对于每对共同活动，检查它们在两个序列中的相对顺序是否一致
    3. 计算顺序一致的活动对比例作为评分
    """
    # 找出两个列表中都存在的活动
    gt_set = set(gt_activities)
    pred_set = set(pred_activities)
    common_activities = gt_set.intersection(pred_set)
    
    # 如果没有共同活动，返回0分
    if len(common_activities) <= 1:
        return 0.0
    
    # 创建活动到索引的映射
    gt_indices = {act: idx for idx, act in enumerate(gt_activities)}
    pred_indices = {act: idx for idx, act in enumerate(pred_activities)}
    
    # 计算所有可能的活动对
    total_pairs = 0
    correct_pairs = 0
    
    # 遍历所有可能的活动对
    activity_list = list(common_activities)
    for i in range(len(activity_list)):
        for j in range(i+1, len(activity_list)):
            act_i = activity_list[i]
            act_j = activity_list[j]
            
            # 在gt中的顺序
            gt_order = gt_indices[act_i] < gt_indices[act_j]
            # 在pred中的顺序
            pred_order = pred_indices[act_i] < pred_indices[act_j]
            
            total_pairs += 1
            if gt_order == pred_order:
                correct_pairs += 1
    
    # 计算顺序一致的比例
    if total_pairs == 0:
        return 0.0
    else:
        return correct_pairs / total_pairs

def calculate_exact_match(gt_activities, pred_activities):
    """计算完全匹配的比例"""
    if len(gt_activities) != len(pred_activities):
        return 0.0
        
    for gt, pred in zip(gt_activities, pred_activities):
        if gt != pred:
            return 0.0
    
    return 1.0

def calculate_activity_recall(gt_activities, pred_activities):
    """计算预测结果包含了多少GT活动"""
    gt_set = set(gt_activities)
    pred_set = set(pred_activities)
    
    if not gt_set:
        return 0.0
        
    common = gt_set.intersection(pred_set)
    return len(common) / len(gt_set)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='internvl8B', help='Model name')
    parser.add_argument('--modality', type=str, default='rgb', help='depth, rgb, ir')

    args = parser.parse_args()

    method = args.method 
    modality = args.modality  # 'depth', 'rgb', 'ir' 
   
    # 读取预测结果CSV文件
    pred_path = f'CUHK-X-VLM/src/sequential_action_recording/predictions/{modality}/pred_{method}.csv'
    data = read_csv_with_activities(pred_path)
    print(f"Loaded {len(data)} samples from {pred_path}")

    # 初始化评分统计
    total_samples = 0
    valid_samples = 0
    total_order_score = 0.0
    total_exact_match = 0.0
    total_recall = 0.0
    
    # 逐行比较顺序一致性
    for item in data:
        path = item["path"]
        logic = item["logic"]
        shuffled = item["shuffled"]
        vlm_result = item["vlm_result"]
        
        # 检查是否有无效结果
        if isinstance(vlm_result, str) and (vlm_result in ["OOM", "video cannot find"] or vlm_result.startswith("ERROR:")):
            print(f"跳过无效样本: {path}")
            continue
            
        # 标准化活动列表，会自动处理<answer>标签
        gt_activities = normalize_activities(logic)
        pred_activities = normalize_activities(vlm_result)
        
        # 打印原始和处理后的预测结果，用于调试
        if isinstance(vlm_result, str) and "<answer>" in vlm_result:
            print(f"原始预测: {vlm_result}")
            print(f"处理后预测: {pred_activities}")
        
        # 计算顺序一致性评分
        order_score = calculate_order_score(gt_activities, pred_activities)
        exact_match = calculate_exact_match(gt_activities, pred_activities)
        recall = calculate_activity_recall(gt_activities, pred_activities)
        
        print(f"\n样本: {path}")
        print(f"GT顺序: {gt_activities}")
        print(f"预测顺序: {pred_activities}")
        print(f"顺序一致性评分: {order_score:.4f}")
        print(f"完全匹配: {'是' if exact_match else '否'}")
        print(f"活动召回率: {recall:.4f}")
        
        total_order_score += order_score
        total_exact_match += exact_match
        total_recall += recall
        valid_samples += 1
        total_samples += 1
    
    # 计算平均分数
    if valid_samples > 0:
        avg_order_score = total_order_score / valid_samples
        avg_exact_match = total_exact_match / valid_samples
        avg_recall = total_recall / valid_samples
        
        print("\n=== 总体评估结果 ===")
        print(f"总样本数: {total_samples}")
        print(f"有效样本数: {valid_samples}")
        print(f"平均顺序一致性评分: {avg_order_score:.4f}")
        print(f"完全匹配比例: {avg_exact_match:.4f}")
        print(f"平均活动召回率: {avg_recall:.4f}")
    else:
        print("没有有效样本进行评估")
    
    # 保存评估结果
    results_dir = f'CUHK-X-VLM/src/sequential_action_recording/scores/{modality}'
    os.makedirs(results_dir, exist_ok=True)
    output_csv = f'{results_dir}/{method}_order_scores.csv'
    with open(output_csv, mode="w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total_Samples", total_samples])
        writer.writerow(["Valid_Samples", valid_samples])
        writer.writerow(["Avg_Order_Score", avg_order_score if valid_samples > 0 else 0])
        writer.writerow(["Exact_Match_Ratio", avg_exact_match if valid_samples > 0 else 0])
        writer.writerow(["Avg_Activity_Recall", avg_recall if valid_samples > 0 else 0])
    
    print(f"\n评估结果已保存到 {output_csv}")