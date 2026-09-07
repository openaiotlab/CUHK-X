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
    data = []
    model_name = os.path.basename(csv_path).split('_')[1].split('.')[0]
    print(f"model name: {model_name}")
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)

            path_idx = header.index("Path") if "Path" in header else 0
            logic_idx = header.index("Logic") if "Logic" in header else 1
            result_idx = header.index("vlm_result") if "vlm_result" in header else 2
            
            for row in reader:
                if len(row) > max(path_idx, logic_idx, result_idx):
                    path = row[path_idx]
                    logic = row[logic_idx]
                    vlm_result = row[result_idx]
                    
                    # Process the output format of the videochatr1 model specially
                    if model_name == "videochatr1" and isinstance(vlm_result, str) and "<answer>" in vlm_result:
                        print(f"Detect answers in videochatr1 format.")
                    
                    data.append({
                        "path": path,
                        "logic": logic,
                        "vlm_result": vlm_result,
                        "model_name": model_name
                    })
    except Exception as e:
        print(f"Error reading CSV file: {e}")
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
        # extract the content between <answer> and </answer>
        activity_str = match.group(1).strip()

    activity_str = activity_str.replace('，', ',')
    activities = [act.strip().lower() for act in activity_str.split(',')]
    activities = [act for act in activities if act]
    return activities

def calculate_activity_precision_recall(gt_activities, pred_activities):
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
    gt_set = set(gt_activities)
    pred_set = set(pred_activities)
    missing = gt_set - pred_set
    return list(missing)

def calculate_extra_activities(gt_activities, pred_activities):
    gt_set = set(gt_activities)
    pred_set = set(pred_activities)
    extra = pred_set - gt_set
    return list(extra)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='internvl2B', help='Model name')
    parser.add_argument('--modality', type=str, default='rgb', help='depth, rgb, ir, thermal')

    args = parser.parse_args()

    method = args.method 
    modality = args.modality

    pred_path = f'CUHK-X-VLM/src/action_selection/predictions/{modality}/pred_{method}.csv'
    data = read_csv_with_activities(pred_path)
    print(f"Loaded {len(data)} samples from {pred_path}")
    print(f"model name: {method}")

    total_samples = 0
    valid_samples = 0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_complete_predictions = 0

    all_gt_activities = set()
    missing_activity_counts = {}

    for item in data:
        path = item["path"]
        gt = item["logic"]
        vlm_result = item["vlm_result"]
        model_name = item.get("model_name", method)

        if vlm_result in ["OOM", "video cannot find"] or (isinstance(vlm_result, str) and vlm_result.startswith("ERROR:")):
            print(f"skip invalid samples: {path}")
            continue

        gt_activities = normalize_activities(gt)
        pred_activities = normalize_activities(vlm_result)

        if isinstance(vlm_result, str) and "<answer>" in vlm_result:
            print(f"vlm_result: {vlm_result}")
            print(f"pred_activities: {pred_activities}")

        all_gt_activities.update(gt_activities)

        precision, recall, f1 = calculate_activity_precision_recall(gt_activities, pred_activities)

        missing_activities = calculate_missing_activities(gt_activities, pred_activities)
        for act in missing_activities:
            missing_activity_counts[act] = missing_activity_counts.get(act, 0) + 1

        extra_activities = calculate_extra_activities(gt_activities, pred_activities)
        
        is_complete = len(missing_activities) == 0
        
        print(f"\npath: {path}")
        print(f"gt_activities: {gt_activities}")
        print(f"pred_activities: {pred_activities}")
        print(f"precision: {precision:.4f}")
        print(f"recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"complete prediction: {'true' if is_complete else 'false'}")
        if missing_activities:
            print(f"missing_activities: {missing_activities}")
        if extra_activities:
            print(f"extra_activities: {extra_activities}")
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        if is_complete:
            total_complete_predictions += 1
        valid_samples += 1
        total_samples += 1

    if valid_samples > 0:
        avg_precision = total_precision / valid_samples
        avg_recall = total_recall / valid_samples
        avg_f1 = total_f1 / valid_samples
        complete_ratio = total_complete_predictions / valid_samples
        
        print("\n=== evaluation scores ===")
        print(f"total_samples: {total_samples}")
        print(f"valid_samples: {valid_samples}")
        print(f"avg_precision: {avg_precision:.4f}")
        print(f"avg_recall: {avg_recall:.4f}")
        print(f"avg_f1: {avg_f1:.4f}")
        print(f"complete_ratio: {complete_ratio:.4f} ({total_complete_predictions}/{valid_samples})")

        print(f"\nThere are {len(all_gt_activities)} kinds of ground truth activities.")

        if missing_activity_counts:
            print("\nmissing_activity_counts:")
            sorted_missing = sorted(missing_activity_counts.items(), key=lambda x: x[1], reverse=True)
            for act, count in sorted_missing:
                miss_rate = count / valid_samples
                print(f"  {act}: {count} (miss_rate: {miss_rate:.2%})")
    else:
        print("There is no ground truth activities.")
        sorted_missing = []

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

    if valid_samples > 0:
        activity_stats_csv = f'{results_dir}/{method}_activity_stats.csv'
        with open(activity_stats_csv, mode="w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Activity", "Missing_Count", "Missing_Rate"])
            for act, count in sorted_missing:
                miss_rate = count / valid_samples
                writer.writerow([act, count, f"{miss_rate:.4f}"])
        
        print(f"Saved in {activity_stats_csv}")
    
    print(f"\nResults have been saved to {output_csv}")