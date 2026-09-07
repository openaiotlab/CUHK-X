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
    data = []
    model_name = os.path.basename(csv_path).split('_')[1].split('.')[0]
    print(f"model_name: {model_name}")
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)

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

                    if model_name == "videochatr1" and isinstance(vlm_result, str) and "<answer>" in vlm_result:
                        print(f"Detect answers of videochatr1 format.")
                        
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
    if not activity_str:
        return []
    
    if not isinstance(activity_str, str):
        activity_str = str(activity_str)

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.search(answer_pattern, activity_str, re.DOTALL)
    if match:
        activity_str = match.group(1).strip()

    activity_str = activity_str.replace('，', ',')

    activities = [act.strip().lower() for act in activity_str.split(',')]
    activities = [act for act in activities if act]
    return activities

def calculate_order_score(gt_activities, pred_activities):
    gt_set = set(gt_activities)
    pred_set = set(pred_activities)
    common_activities = gt_set.intersection(pred_set)

    if len(common_activities) <= 1:
        return 0.0

    gt_indices = {act: idx for idx, act in enumerate(gt_activities)}
    pred_indices = {act: idx for idx, act in enumerate(pred_activities)}

    total_pairs = 0
    correct_pairs = 0

    activity_list = list(common_activities)
    for i in range(len(activity_list)):
        for j in range(i+1, len(activity_list)):
            act_i = activity_list[i]
            act_j = activity_list[j]

            gt_order = gt_indices[act_i] < gt_indices[act_j]
            pred_order = pred_indices[act_i] < pred_indices[act_j]
            
            total_pairs += 1
            if gt_order == pred_order:
                correct_pairs += 1

    if total_pairs == 0:
        return 0.0
    else:
        return correct_pairs / total_pairs

def calculate_exact_match(gt_activities, pred_activities):
    if len(gt_activities) != len(pred_activities):
        return 0.0
        
    for gt, pred in zip(gt_activities, pred_activities):
        if gt != pred:
            return 0.0
    
    return 1.0

def calculate_activity_recall(gt_activities, pred_activities):
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

    pred_path = f'CUHK-X-VLM/src/sequential_action_recording/predictions/{modality}/pred_{method}.csv'
    data = read_csv_with_activities(pred_path)
    print(f"Loaded {len(data)} samples from {pred_path}")

    total_samples = 0
    valid_samples = 0
    total_order_score = 0.0
    total_exact_match = 0.0
    total_recall = 0.0

    for item in data:
        path = item["path"]
        logic = item["logic"]
        shuffled = item["shuffled"]
        vlm_result = item["vlm_result"]

        if isinstance(vlm_result, str) and (vlm_result in ["OOM", "video cannot find"] or vlm_result.startswith("ERROR:")):
            print(f"Skip invalid samples: {path}")
            continue

        gt_activities = normalize_activities(logic)
        pred_activities = normalize_activities(vlm_result)

        if isinstance(vlm_result, str) and "<answer>" in vlm_result:
            print(f"vlm_result: {vlm_result}")
            print(f"pred_activities: {pred_activities}")

        order_score = calculate_order_score(gt_activities, pred_activities)
        exact_match = calculate_exact_match(gt_activities, pred_activities)
        recall = calculate_activity_recall(gt_activities, pred_activities)
        
        print(f"\npath: {path}")
        print(f"gt_activities: {gt_activities}")
        print(f"pred_activities: {pred_activities}")
        print(f"order_score: {order_score:.4f}")
        print(f"is match: {'true' if exact_match else 'false'}")
        print(f"recall: {recall:.4f}")
        
        total_order_score += order_score
        total_exact_match += exact_match
        total_recall += recall
        valid_samples += 1
        total_samples += 1

    if valid_samples > 0:
        avg_order_score = total_order_score / valid_samples
        avg_exact_match = total_exact_match / valid_samples
        avg_recall = total_recall / valid_samples
        
        print("\n=== evaluation scores ===")
        print(f"total_samples: {total_samples}")
        print(f"valid_samples: {valid_samples}")
        print(f"avg_order_score: {avg_order_score:.4f}")
        print(f"avg_exact_match: {avg_exact_match:.4f}")
        print(f"avg_recall: {avg_recall:.4f}")
    else:
        print("There are no valid samples.")

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
    
    print(f"\nResults have been saved to {output_csv}")