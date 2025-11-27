import os
import csv
import pandas as pd
from bert_score import score
from rouge import Rouge  
import argparse
import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

def read_csv_to_dict(csv_path):
    """
    Read a csv file and return a dict: {video_path: caption}
    """
    data = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                data[row[0]] = row[1]
    return data

def read_xlsx_to_dict(csv_path):
    """
    Read a CSV file (in the same format as the original Excel) and return a dict: {video_path: caption}
    Adds a fixed base path to each video path.
    """
    data = {}
    try:
        # 读取CSV文件（假设与Excel结构相同：第一列是视频路径，第二列是字幕）
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        if len(df.columns) >= 2:
            for index, row in df.iterrows():
                # 在视频路径前添加基础路径
                video_path = row.iloc[0]
                data[video_path] = row.iloc[1]
        else:
            print(f"CSV文件 {csv_path} 格式不正确，至少需要两列")
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")

    return data


def calculate_bleu(pred_list, gt_list):
    # 使用平滑函数避免零分问题
    smoothing = SmoothingFunction().method1
    
    bleu_scores = {
        'bleu-1': 0.0,
        'bleu-2': 0.0,
        'bleu-3': 0.0,
        'bleu-4': 0.0
    }
    
    valid_pairs = 0
    
    for pred, gt in zip(pred_list, gt_list):
        if not pred or not gt:
            continue
            
        try:
            # 分词
            reference = nltk.word_tokenize(gt.lower())
            hypothesis = nltk.word_tokenize(pred.lower())
            
            # 空序列检查
            if len(hypothesis) == 0 or len(reference) == 0:
                continue
                
            # 计算各个级别的BLEU分数
            bleu_1 = sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu_2 = sentence_bleu([reference], hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu_3 = sentence_bleu([reference], hypothesis, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
            bleu_4 = sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
            
            bleu_scores['bleu-1'] += bleu_1
            bleu_scores['bleu-2'] += bleu_2
            bleu_scores['bleu-3'] += bleu_3
            bleu_scores['bleu-4'] += bleu_4
            
            valid_pairs += 1
        except Exception as e:
            print(f"Error calculating BLEU for a pair: {e}")
    
    # 计算平均值
    if valid_pairs > 0:
        for k in bleu_scores:
            bleu_scores[k] /= valid_pairs
    
    return bleu_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='qwenvl7B', help='Model name')
    parser.add_argument('--modality', type=str, default='rgb', help='depth, rgb, ir, thermal')

    args = parser.parse_args()

    method = args.method 
    modality = args.modality  # 'depth', 'rgb', 'ir' 
   
    pred_path = f'CUHK-X-VLM/src/task_caption/predictions/{modality}/pred_{method}.csv'
    if modality == 'rgb':
        gt_path = 'GT_folder/LM_RGB_GT.csv'
    if modality == 'ir':
        gt_path = 'GT_folder/LM_IR_GT.csv'
    if modality == 'depth':
        gt_path = 'GT_folder/LM_Depth_GT.csv'
    if modality == 'thermal':
        gt_path = 'GT_folder/LM_Thermal_GT.csv'
    pred_dict = read_csv_to_dict(pred_path)
    gt_dict = read_xlsx_to_dict(gt_path)  # 使用新函数读取xlsx文件

    # Find common video paths
    common_keys = list(set(pred_dict.keys()) & set(gt_dict.keys()))
    print(f"Total matched samples: {len(common_keys)}")

    # 筛选出不是OOM的样本
    valid_keys = []
    for k in common_keys:
        if pred_dict[k] != "OOM" and pred_dict[k] != "video cannot find" and not pred_dict[k].startswith("ERROR:"):
            valid_keys.append(k)
    print(f"Total valid samples (excluding OOM and errors): {len(valid_keys)}")
    print(f"Skipped {len(common_keys) - len(valid_keys)} invalid samples")

    pred_list = [pred_dict[k] for k in valid_keys]
    gt_list = [gt_dict[k] for k in valid_keys]

    # Compute BERTScore
    P, R, F1 = score(pred_list, gt_list, lang="en", verbose=True)
    avg_P = P.mean().item()
    avg_R = R.mean().item()
    avg_F1 = F1.mean().item()
    print(f"Average BERTScore P: {avg_P:.4f}")
    print(f"Average BERTScore R: {avg_R:.4f}")
    print(f"Average BERTScore F1: {avg_F1:.4f}")

    # Compute METEOR scores
    if not nltk.data.find('tokenizers/punkt'):
        nltk.download(['punkt', 'wordnet', 'omw-1.4'])
    meteor_scores = []
    for pred, gt in zip(pred_list, gt_list):
        # Tokenize both strings
        pred_tokens = nltk.word_tokenize(pred.lower())
        gt_tokens = nltk.word_tokenize(gt.lower())

        # Compute METEOR score with tokenized inputs
        score_val = meteor_score([gt_tokens], pred_tokens)
        meteor_scores.append(score_val)

    # Compute average METEOR
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if len(meteor_scores) > 0 else 0.0
    print(f"\nMETEOR: {avg_meteor:.4f}")



    # Compute ROUGE scores
    rouge = Rouge()
    rouge_scores = {"rouge-1": {"f": 0, "p": 0, "r": 0}, 
                    "rouge-2": {"f": 0, "p": 0, "r": 0}, 
                    "rouge-l": {"f": 0, "p": 0, "r": 0}}
    
    valid_pairs = 0
    for pred, gt in zip(pred_list, gt_list):
        try:
            # 确保文本不为空
            if not pred or not gt:
                continue
                
            # 有些文本可能会导致ROUGE计算错误，使用try-except处理
            scores = rouge.get_scores(pred, gt)[0]
            for k in rouge_scores:
                for metric in rouge_scores[k]:
                    rouge_scores[k][metric] += scores[k][metric]
            valid_pairs += 1
        except Exception as e:
            print(f"Error calculating ROUGE for a pair: {e}")
    
    # 计算平均值
    if valid_pairs > 0:
        for k in rouge_scores:
            for metric in rouge_scores[k]:
                rouge_scores[k][metric] /= valid_pairs
    
    print("\nROUGE Scores:")
    print(f"ROUGE-1 F1: {rouge_scores['rouge-1']['f']:.4f}")
    print(f"ROUGE-2 F1: {rouge_scores['rouge-2']['f']:.4f}")
    print(f"ROUGE-L F1: {rouge_scores['rouge-l']['f']:.4f}")
    
    # 计算BLEU分数
    bleu_scores = calculate_bleu(pred_list, gt_list)
    print("\nBLEU Scores:")
    print(f"BLEU-1: {bleu_scores['bleu-1']:.4f}")
    print(f"BLEU-2: {bleu_scores['bleu-2']:.4f}")
    print(f"BLEU-3: {bleu_scores['bleu-3']:.4f}")
    print(f"BLEU-4: {bleu_scores['bleu-4']:.4f}")

    # 确保输出目录存在
    os.makedirs(f'CUHK-X-VLM/src/task_caption/scores/{modality}', exist_ok=True)
    
    # save results to a new CSV file
    output_csv = f'CUHK-X-VLM/src/task_caption/scores/{modality}/{method}_caption.csv'
    with open(output_csv, mode="w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["BERTScore_P", avg_P])
        writer.writerow(["BERTScore_R", avg_R])
        writer.writerow(["BERTScore_F1", avg_F1])
        writer.writerow(["ROUGE-1_F1", rouge_scores['rouge-1']['f']])
        writer.writerow(["ROUGE-2_F1", rouge_scores['rouge-2']['f']])
        writer.writerow(["ROUGE-L_F1", rouge_scores['rouge-l']['f']])
        writer.writerow(["BLEU-1", bleu_scores['bleu-1']])
        writer.writerow(["BLEU-2", bleu_scores['bleu-2']])
        writer.writerow(["BLEU-3", bleu_scores['bleu-3']])
        writer.writerow(["BLEU-4", bleu_scores['bleu-4']])
        writer.writerow(["METEOR", avg_meteor])
    print(f"BERTScore, ROUGE and BLEU results have been saved to {output_csv}")