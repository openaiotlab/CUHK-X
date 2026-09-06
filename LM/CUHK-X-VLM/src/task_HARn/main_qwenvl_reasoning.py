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
                    logic = row[1]
                    candidate = row[2]
                    data.append([path, logic, candidate])
            
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        traceback.print_exc()
        
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default='7B', help='Model size: 7B or 3B')
    parser.add_argument('--modality', type=str, default='rgb', help='depth, rgb, ir')
    args = parser.parse_args()

    model_size = args.model_size
    modality = args.modality

    if modality == 'rgb':
        test_csv_path = 'GT_folder/SM_RGB_logic.csv'
    if modality == 'ir':
        test_csv_path = 'GT_folder/SM_IR_logic.csv'
    if modality == 'depth':
        test_csv_path = 'GT_folder/SM_Depth_logic.csv'
    if modality == 'thermal':
        test_csv_path = ''

    test_data = read_csv_file(test_csv_path)
    print(f"Loaded {len(test_data)} samples from {test_csv_path}")

    output_dir = f'CUHK-X-VLM/src/task_logic/predictions/{modality}'
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
        print(f"The {start_idx}th sample has been processed; we will continue from the {start_idx+1}th sample.")

    # initialize vlm
    model_path = f"Models/Qwen2.5-VL-{model_size}-Instruct"
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")

    prompt = "Question: What activity is the person likely to do next? Options: "
    # results, idx = [], 1
    idx = 1
    for i, row in enumerate(test_data):
        if i < start_idx:
            continue
        print(row)
        
        if idx >= 3000:
            print("100 samples have been processed, stopping.")
            break 

        video_path = row[0]
        logic = row[1]
        candidate = row[2]
        
        candidate_options = candidate.split(',') if candidate else []
        option_a = candidate_options[0].strip() if len(candidate_options) > 0 else ""
        option_b = candidate_options[1].strip() if len(candidate_options) > 1 else ""

        print(f"Row {i+1}:")
        print(f"  Path: {video_path}")
        print(f"  Logic: {logic}")
        print(f"  Option A: {option_a}")
        print(f"  Option B: {option_b}")
        print("-" * 50)
    
        if option_b:
            choices = '(A) ' + logic + ' (B) ' + option_a + ' (C) ' + option_b
            query = prompt + '\n' + choices + ' \nPlease answer with A, B, or C.'
        else:
            choices = '(A) ' + logic + ' (B) ' + option_a
            query = prompt + '\n' + choices + ' \nPlease answer with A or B.'
        
        print(f"Query: {query}")

        try:
            res = qwenvl_inference(video_path, query, model, processor)
            print(video_path)
            print(res)
        except torch.cuda.OutOfMemoryError:
            # Handle CUDA out of memory error
            print(f"CUDA Out of Memory error for video: {video_path}")
            res = "OOM"  # Out of Memory
        except KeyError as e:
            # Handle KeyError like 'video_fps'
            error_msg = traceback.format_exc()
            print(f"KeyError for video: {video_path}")
            print(error_msg)
            res = f"ERROR: {str(e)}"
        except Exception as e:
            # Handle any other exceptions
            error_msg = traceback.format_exc()
            print(f"Exception for video: {video_path}")
            print(error_msg)
            res = f"ERROR: {str(e)}"    

        print("Path: ", video_path)
        print("Predictions: ", res)

        results.append([video_path, logic, res])
        idx += 1

        # save results
        with open(output_csv, mode="w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Path", "Logic", "vlm_result"])
            writer.writerows(results)
        print(f"Results have been saved to {output_csv}")