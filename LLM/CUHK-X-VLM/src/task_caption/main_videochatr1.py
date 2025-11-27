import os
import csv
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import videochatr1_inference
import traceback
import torch
import gc
import argparse


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
    parser.add_argument('--modality', type=str, default='rgb', help='depth, rgb, ir')
    args = parser.parse_args()
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
        raise ValueError("Invalid modality. Choose from 'depth', 'rgb', 'ir', or 'thermal'.")

    
    file_list = os.listdir(data_dir)

    output_dir = f"CUHK-X-VLM/src/task_caption/predictions/{modality}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_csv = output_dir + f'/pred_videochatr1.csv'

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
                if len(row) >= 2:
                    video_path, res = row[0], row[1]
                    processed_videos.append(video_path)
                    results.append([video_path, res])
        
        start_idx = len(processed_videos) + 1
        print(f"Already processed {start_idx - 1} samples. Will continue from sample #{start_idx}")
        
        if len(processed_videos) >= process_num:
            print(f"Already processed {len(processed_videos)} samples, which meets or exceeds target of {process_num}.")
            exit(0)

    # Initialize VLM
    model_path = "Models/VideoChat-R1_7B"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    prompt = "Describe what the person in the video is doing. You can briefly mention the background or setting, but focus mainly on understanding the person's actions."

    total_num = calculate_sample_num(data_dir)
    idx = start_idx
    samples_processed = len(processed_videos)
    
    for s in file_list:
        if samples_processed >= process_num:
            break
            
        data_dir_s = os.path.join(data_dir, s)
        if not os.path.isdir(data_dir_s):
            continue
        file_list_s1 = os.listdir(data_dir_s)
        for s1 in file_list_s1:
            if samples_processed >= process_num:
                break
                
            data_dir_s1 = os.path.join(data_dir_s, s1)
            if modality == 'thermal':
                video_path = os.path.join(data_dir_s1, "Thermal.mp4")
            elif modality == 'depth':
                video_path = os.path.join(data_dir_s1, "Depth.mp4")
            elif modality == 'rgb':
                video_path = os.path.join(data_dir_s1, "RGB.mp4")
            elif modality == 'ir':
                video_path = os.path.join(data_dir_s1, "IR.mp4")
            else:
                raise ValueError("Invalid modality.")

            if video_path in processed_videos:
                idx += 1
                continue

            print(f"Processing sample {idx}/{total_num} (Sample {samples_processed+1}/{process_num}): {video_path}")
                    
            if not os.path.exists(video_path):
                res = "video not found"
            else:
                torch.cuda.empty_cache()
                gc.collect()
                try:
                    res = videochatr1_inference(video_path, prompt, model, processor)
                    print(video_path)
                    print(res)
                except torch.cuda.OutOfMemoryError:
                    print(f"CUDA Out of Memory for video: {video_path}")
                    res = "OOM"
                except KeyError as e:
                    print(f"KeyError for video: {video_path}")
                    print(traceback.format_exc())
                    res = f"ERROR: {str(e)}"
                except Exception as e:
                    print(f"Exception for video: {video_path}")
                    print(traceback.format_exc())
                    res = f"ERROR: {str(e)}"

            results.append([video_path, res])
            samples_processed += 1
            idx += 1

            # save results
            with open(output_csv, mode="w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["video_path", "vlm_result"])
                writer.writerows(results)
            print(f"Results have been saved to {output_csv}")

            if samples_processed >= process_num:
                print(f"Reached target of {process_num} samples. Exiting.")
                break