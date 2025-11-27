import os
import csv
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import AutoModel, AutoTokenizer
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import internvl_inference
import torch
import argparse
import traceback
import random
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
                    caption = row[1]
                    gt = row[2]
                    data.append([path, caption, gt])
                else:
                    print(f"警告: 行数据不完整: {row}")
            
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        traceback.print_exc()
        
    return data


def shuffle_activities(activity_str):
    """
    将逗号分隔的活动列表随机打乱顺序
    
    Args:
        activity_str: 逗号分隔的活动字符串，如 "wiping hands,combing hair,getting dressed"
        
    Returns:
        打乱后的逗号分隔字符串
    """
    if not activity_str:
        return ""
    
    # 按逗号分割活动
    activities = [act.strip() for act in activity_str.split(',')]
    
    # 打印原始顺序
    print(f"原始活动顺序: {activities}")
    
    # 随机打乱顺序
    random.shuffle(activities)
    
    # 打印打乱后的顺序
    print(f"打乱后活动顺序: {activities}")
    
    # 重新组合成字符串
    shuffled_str = ', '.join(activities)
    
    return shuffled_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='internvl', help='Model name')
    parser.add_argument('--model_size', type=str, default='2B', help='Model size: 7B or 3B')
    parser.add_argument('--modality', type=str, default='rgb', help='depth, rgb, ir')
    args = parser.parse_args()

    model = args.model
    model_size = args.model_size  # or '8B', '2B'
    modality = args.modality  # 'depth', 'rgb', 'ir'

    if modality == 'rgb':
        test_csv_path = 'GT_folder/LM_RGB_sequential.csv'
    elif modality == 'ir':
        test_csv_path = 'GT_folder/LM_IR_sequential.csv'
    elif modality == 'depth':
        test_csv_path = 'GT_folder/LM_Depth_sequential.csv'
    elif modality == 'thermal':
        test_csv_path = 'GT_folder/LM_Thermal_sequential.csv'

    test_data = read_csv_file(test_csv_path)
    print(f"Loaded {len(test_data)} samples from {test_csv_path}")

    output_dir = f"CUHK-X-VLM/src/sequential_action_recording/predictions/{modality}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_csv = output_dir + f'/pred_internvl{model_size}.csv'
    
    # 检查是否已有输出文件并加载已处理的结果
    results = []
    processed_paths = []
    start_idx = 0
    if os.path.exists(output_csv):
        print(f"找到已有的输出文件: {output_csv}")
        with open(output_csv, mode="r", newline='') as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header
            for row in reader:
                if len(row) >= 3:  # we only save 3 columns
                    path = row[0]
                    processed_paths.append(path)
                    results.append(row)
        start_idx = len(processed_paths)
        print(f"已经处理了 {start_idx} 个样本，将从第 {start_idx+1} 个样本继续")

    # initialize VLM
    model_path = f"Models/InternVL2-{model_size}"
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    base_prompt = f"Question: Please sort the following activity lists in chronological order based on the video content. Output only the activities in this format: wiping hands, combing hair, getting dressed, squats, jumping jacks. Do not include any additional content. Activity lists are: "

    idx = 1
    processed_count = 0 

    for i, row in enumerate(test_data):
        if i < start_idx:
            continue

        video_path = row[0]
        gt = row[2]
        
        print(f"Row {i+1}:")
        print(f"  Path: {video_path}")
        print(f"  GT: {gt}")
        print("-" * 50)

        # shuffle activities
        shuffled_activities = shuffle_activities(gt)
        prompt = base_prompt + shuffled_activities
        print(f"Prompt: {prompt}")

        try:
            res = internvl_inference(video_path, prompt, model, tokenizer)
            print(f"Prediction: {res}")
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
        processed_count += 1  # 增加计数器

        # save results
        with open(output_csv, mode="w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Path", "Logic", "vlm_result"])
            writer.writerows(results)
        print(f"Results have been saved to {output_csv}")





    
