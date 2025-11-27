import os
import csv
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import videollava_inference
import torch
import argparse
import traceback

def read_csv_file(csv_path):
    """
    Read a CSV file and return the content as a list of rows.
    """
    data = []
    try:
        # 读取CSV文件
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # 跳过表头
            for row in reader:
                if len(row) >= 3:  # 确保至少有3列
                    # 确保路径前添加基础路径
                    path = row[0]
                    logic = row[1]
                    candidate = row[2]
                    data.append([path, logic, candidate])
                else:
                    print(f"警告: 行数据不完整: {row}")
            
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        traceback.print_exc()
        
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default='rgb', help='depth, rgb, ir')

    args = parser.parse_args()
    modality = args.modality  # 'depth', 'rgb', 'ir'

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
    output_csv = output_dir + f'/pred_videollava.csv'
    
    # 检查是否已有输出文件并加载已处理的结果
    results = []
    processed_paths = []
    start_idx = 0
    if os.path.exists(output_csv):
        print(f"找到已有的输出文件: {output_csv}")
        with open(output_csv, mode="r", newline='') as f:
            reader = csv.reader(f)
            header = next(reader)  # 跳过表头
            for row in reader:
                if len(row) >= 5:  # 确保行有足够的元素
                    path = row[0]
                    processed_paths.append(path)
                    results.append(row)
        
        start_idx = len(processed_paths)
        print(f"已经处理了 {start_idx} 个样本，将从第 {start_idx+1} 个样本继续")

    # initialize vlm
    model_path = "Models/Video-LLaVA-7B-hf"
    model = VideoLlavaForConditionalGeneration.from_pretrained(model_path)
    processor = VideoLlavaProcessor.from_pretrained(model_path)

    prompt = "Question: What activity is the person likely to do next? Options: "
    # results, idx = [], 1
    idx = 1
    for i, row in enumerate(test_data):
        if i < start_idx:  # 跳过已处理的样本
            continue
        print(row)

        if idx >= 3000:  # 添加100个样本的限制
            print("已处理100个样本，停止处理")
            break 
                
        video_path = row[0]
        logic = row[1]
        candidate = row[2]

        # 处理 candidate 可能为空或只有一个选项的情况
        candidate_options = candidate.split(',') if candidate else []
        option_a = candidate_options[0].strip() if len(candidate_options) > 0 else ""
        option_b = candidate_options[1].strip() if len(candidate_options) > 1 else ""


        print(f"Row {i+1}:")
        print(f"  Path: {video_path}")
        print(f"  Logic: {logic}")
        print(f"  Opions: {option_a}")
        print(f"  Opions: {option_b}")
        print("-" * 50)
    
        # 根据是否有选项 B 来构建查询
        if option_b:
            choices = '(A) ' + logic + ' (B) ' + option_a + ' (C) ' + option_b
            query = prompt + '\n' + choices + ' \nPlease answer with A, B, or C.'
        else:
            # 如果没有选项 B，只提供选项 A 和 Logic
            choices = '(A) ' + logic + ' (B) ' + option_a
            query = prompt + '\n' + choices + ' \nPlease answer with A or B.'
        
        print(f"  Query: {query}")

        try:
            query0 = "USER: <video>"+ query + " ASSISTANT:"
            res = videollava_inference(video_path, query0, model, processor)
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
