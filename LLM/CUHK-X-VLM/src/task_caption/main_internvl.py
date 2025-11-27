import os
import csv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import internvl_inference
import torch
import gc
import argparse
from transformers import AutoModel, AutoTokenizer
import traceback

def calculate_sample_num(data_dir):
    """
    Calculate the total number of samples in the dataset.
    """
    video_paths = []
    for s in file_list:
        data_dir_s = os.path.join(data_dir, s)
        if not os.path.isdir(data_dir_s):
            continue
        file_list_s1 = os.listdir(data_dir_s)
        for s1 in file_list_s1:
            data_dir_s1 = os.path.join(data_dir_s, s1)
            video_path = os.path.join(data_dir_s1, "RGB.mp4")
            video_paths.append(video_path)
    total = len(video_paths)
    print(f"Total samples: {total}")
    return total



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='internvl', help='Model name')
    parser.add_argument('--model_size', type=str, default='8B', help='Model size: 7B or 3B')
    parser.add_argument('--modality', type=str, default='rgb', help='depth, rgb, ir')
    args = parser.parse_args()

    model = args.model 
    model_size = args.model_size  # or '8B', '2B'
    modality = args.modality  # 'depth', 'rgb', 'ir'
    
    process_num = 2000  # 处理的样本数量限制

    if modality == 'thermal':
        print("Using thermal modality")
        data_dir = "LM_video/Thermal"
    elif modality == 'rgb':
        print("Using RGB modality")
        data_dir = "LM_video/RGB"
    elif modality == 'ir':
        print("Using IR modality")
        data_dir = "LM_video/IR"
    elif modality == 'depth':
        print("Using Depth modality")
        data_dir = "LM_video/Depth"
    else:
        raise ValueError("Invalid modality. Choose from 'depth', 'rgb', 'thermal' or 'ir'.")

    file_list = os.listdir(data_dir)

    output_dir = f"CUHK-X-VLM/src/task_caption/predictions/{modality}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_csv = output_dir +  f'/pred_internvl{model_size}.csv'


    # Check if output file exists and load processed results
    processed_videos = []
    results = []
    start_idx = 1
    
    if os.path.exists(output_csv):
        print(f"Found existing output file: {output_csv}")
        with open(output_csv, mode="r", newline='') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:  # Ensure the row has at least two elements
                    video_path, res = row[0], row[1]
                    processed_videos.append(video_path)
                    results.append([video_path, res])
        
        start_idx = len(processed_videos) + 1
        print(f"Already processed {start_idx - 1} samples. Will continue from sample #{start_idx}")
        
        # 检查是否已经处理了足够的样本
        if len(processed_videos) >= process_num:
            print(f"Already processed {len(processed_videos)} samples, which meets or exceeds the target of {process_num}.")
            exit(0)


    # initialize vlm
    model_path = f"Models/InternVL2-{model_size}"
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    # prompt = "Describe the video content."
    prompt = "Describe what the person in the video is doing. You can briefly mention the background or setting, but focus mainly on understanding the person's actions."

    total_num = calculate_sample_num(data_dir) # collect total number of video paths (used for progress tracking)
    # results, idx = [], 1

    idx = 1
    samples_processed = len(processed_videos)  # 已处理的样本数量
    
    for s in file_list:
        if samples_processed >= process_num:
            break
            
        data_dir_s = os.path.join(data_dir, s)
        if not os.path.isdir(data_dir_s):   # <-- skip non-directories
            continue
        file_list_s1 = os.listdir(data_dir_s)
        for s1 in file_list_s1:
            if samples_processed >= process_num:
                break
                
            data_dir_s1 = os.path.join(data_dir_s, s1)
            # print(data_dir_s1)
            file_list_s2 = os.listdir(data_dir_s1)
            if modality == 'thermal':
                video_path = os.path.join(data_dir_s1, "Thermal.mp4")
            elif modality == 'depth':
                video_path = os.path.join(data_dir_s1, "Depth.mp4")
            elif modality == 'rgb':
                video_path = os.path.join(data_dir_s1, "RGB.mp4")
            elif modality == 'ir':
                video_path = os.path.join(data_dir_s1, "IR.mp4")
            else:
                raise ValueError("Invalid modality. Choose from 'depth', 'rgb', or 'ir'.")


            # Skip already processed videos
            if video_path in processed_videos:
                idx += 1
                continue

            print(f"Processing sample {idx}/{total_num} (Sample {samples_processed+1}/{process_num}): {video_path}")

            # VLM inference
            if not os.path.exists(video_path):
                res = "video cannot find"
            else:
                torch.cuda.empty_cache()
                gc.collect()
                try:
                    res = internvl_inference(video_path, prompt, model, tokenizer)
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
                    
            results.append([video_path, res])
            samples_processed += 1  # 更新已处理样本计数
            idx += 1

            # save results
            with open(output_csv, mode="w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["video_path", "vlm_result"])
                writer.writerows(results)
            print(f"Results have been saved to {output_csv}")
            
            # check if there is enough samples
            if samples_processed >= process_num:
                print(f"Reached target of {process_num} samples. Exiting.")
                break
    
    print(f"Completed processing {samples_processed} samples")