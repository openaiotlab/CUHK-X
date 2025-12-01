import os
import csv
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import qwenvl_inference
import torch
import argparse
import traceback

def read_csv_file(csv_path):
    data = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) >= 3:
                    path = row[0]
                    caption = row[1]
                    gt = row[2]
                    data.append([path, caption, gt])
            
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        traceback.print_exc()
        
    return data


def read_class_names(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
            class_names_str = ', '.join(class_names)
            return class_names_str
    except Exception as e:
        print(e)
        traceback.print_exc()
        return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default='7B', help='Model size: 7B or 3B')
    parser.add_argument('--modality', type=str, default='rgb', help='depth, rgb, ir, thermal')
    args = parser.parse_args()

    model_size = args.model_size
    modality = args.modality

    if modality == 'rgb':
        test_csv_path = 'GT_folder/LM_RGB_sequential.csv'
    elif modality == 'ir':
        test_csv_path = 'GT_folder/LM_IR_sequential.csv'
    elif modality == 'depth':
        test_csv_path = 'GT_folder/LM_Depth_sequential.csv'
    elif modality == 'thermal':
        test_csv_path = 'GT_folder/LM_Thermal_sequential.csv'

    class_names_file = 'class_names.txt'
    class_names_str = read_class_names(class_names_file)
    print(f"class names list: {class_names_str}")

    test_data = read_csv_file(test_csv_path)
    print(f"Loaded {len(test_data)} samples from {test_csv_path}")

    output_dir = f"CUHK-X-VLM/src/action_selection/predictions/{modality}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_csv = output_dir + f'/pred_qwenvl{model_size}.csv'

    results = []
    processed_paths = []
    start_idx = 0
    if os.path.exists(output_csv):
        with open(output_csv, mode="r", newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) >= 3:
                    path = row[0]
                    processed_paths.append(path)
                    results.append(row)
        
        start_idx = len(processed_paths)
        print(f"已经处理了 {start_idx} 个样本，将从第 {start_idx+1} 个样本继续")

    # initialize vlm
    model_path = f"Models/Qwen2.5-VL-{model_size}-Instruct"
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")

    prompt = f"Question: What activity is the person performing in the video? You must choose only from the following activities: {class_names_str}. You can choose multiple activities if necessary. \nPlease answer with the activity name or names, separated by commas such as standing up, walking, mopping, walking, etc."

    # results, idx = [], 1
    idx = 1
    processed_count = 0 
    for i, row in enumerate(test_data):
        if i < start_idx:  # 跳过已处理的样本
            continue
        
        print(row)
        video_path = row[0]
        gt = row[2]
        
        print(f"Row {i+1}:")
        print(f"Path: {video_path}")
        print(f"GT: {gt}")
        print("-" * 50)

        try:
            res = qwenvl_inference(video_path, prompt, model, processor)
            print(video_path)
            print(res)
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA Out of Memory error for video: {video_path}")
            res = "OOM"
        except KeyError as e:
            error_msg = traceback.format_exc()
            print(f"KeyError for video: {video_path}")
            print(error_msg)
            res = f"ERROR: {str(e)}"
        except Exception as e:
            error_msg = traceback.format_exc()
            print(f"Exception for video: {video_path}")
            print(error_msg)
            res = f"ERROR: {str(e)}"    

        print("Path: ", video_path)
        print("Predictions: ", res)

        results.append([video_path, gt, res])
        idx += 1
        processed_count += 1

        # save results
        with open(output_csv, mode="w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Path", "Logic", "vlm_result"])
            writer.writerows(results)
        print(f"Results have been saved to {output_csv}")