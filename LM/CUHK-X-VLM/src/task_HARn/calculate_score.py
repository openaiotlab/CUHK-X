import csv
import os
import re
import argparse

def calculate_accuracy(csv_path, method):
    total_samples = 0
    correct_samples = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 3:  # Ensure row has enough columns
                if row[2].strip() == 'OOM':  
                    continue  
                result = row[2].strip()
                
                if method == 'videochatr1':
                    match = re.search(r'<answer>([A-C])</answer>', result)
                    if match:
                        result = match.group(1)
                    else:
                        match = re.search(r'\b([A-C])\b', result)
                        if match:
                            result = match.group(1)

                total_samples += 1

                is_correct = False

                if result == 'A':
                    is_correct = True
                elif re.search(r'\(A\)', result):
                    is_correct = True
                elif re.match(r'^A\s', result):
                    is_correct = True
                elif re.search(r'[答案选择是]\s*A', result):
                    is_correct = True
                
                if is_correct:
                    correct_samples += 1
                    print(f"Correct answer found: {result}")
                else:
                    print(f"Incorrect answer: {result}")
    
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