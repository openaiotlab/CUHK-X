import csv
import os
import re
import argparse

def calculate_accuracy(csv_path, method):
    """
    Read CSV file and calculate the proportion of samples with vlm_result as 'A'
    """
    total_samples = 0
    correct_samples = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            # print(f"Processing row: {row}")  # Debugging line to see the row being processed
            if len(row) >= 3:  # Ensure row has enough columns
                if row[2].strip() == 'OOM':  
                    continue  
                result = row[2].strip()
                
                # 处理videochatr1的特殊格式 <answer>X</answer>
                if method == 'videochatr1':
                    match = re.search(r'<answer>([A-C])</answer>', result)
                    if match:
                        result = match.group(1)  # 提取<answer>和</answer>之间的内容
                    else:
                        # 如果没有找到特定格式，尝试直接找A、B或C
                        match = re.search(r'\b([A-C])\b', result)
                        if match:
                            result = match.group(1)

                total_samples += 1
                
                # 检查多种格式的A答案
                is_correct = False
                
                # 1. 直接是A
                if result == 'A':
                    is_correct = True
                # 2. 包含(A)格式
                elif re.search(r'\(A\)', result):
                    is_correct = True
                # 3. 以A开头后跟空格或其他字符（如 "A typing on a keyboard"）
                elif re.match(r'^A\s', result):
                    is_correct = True
                # 4. 包含"答案是A"、"选择A"等中文表述
                elif re.search(r'[答案选择是]\s*A', result):
                    is_correct = True
                
                if is_correct:
                    correct_samples += 1
                    print(f"Correct answer found: {result}")  # 调试输出
                else:
                    print(f"Incorrect answer: {result}")  # 调试输出
    
    accuracy = correct_samples / total_samples if total_samples > 0 else 0
    return accuracy, correct_samples, total_samples

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='videochatr1', help='Model name')
    parser.add_argument('--modality', type=str, default='ir', help='depth, rgb, ir')
    args = parser.parse_args()

    method = args.method 
    modality = args.modality  # 'depth', 'rgb', 'ir' 
   
    csv_path = f"src/task_logic/predictions/{modality}/pred_{method}.csv"
    print(f"Calculating accuracy for {csv_path}...")
    
    # Calculate accuracy - 传入method参数
    accuracy, correct, total = calculate_accuracy(csv_path, method)
    
    print(f"Total samples: {total}")
    print(f"Samples with answer 'A': {correct}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total}, {accuracy*100:.2f}%)")

    # Create directory if it doesn't exist
    os.makedirs(f"src/task_logic/scores/{modality}", exist_ok=True)
    
    # Save accuracy to CSV file
    output_csv = f"src/task_logic/scores/{modality}/{method}_accuracy.csv"
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Accuracy", "Correct", "Total"])
        writer.writerow([method, accuracy, correct, total])
    
    print(f"Accuracy has been saved to {output_csv}")