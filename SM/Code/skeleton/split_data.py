import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split

root_dir = 'YOUR/PATH/TO/CUHKX/SM/Data/Skeleton'
save_dir = 'YOUR/PATH/TO/CUHKX/SM/Code/skeleton/split_data_results'
os.makedirs(save_dir, exist_ok=True)

available_fonts = [f.name for f in fm.fontManager.ttflist]
chinese_fonts = [name for name in available_fonts if any(chinese in name for chinese in ['WenQuanYi', 'Noto Sans CJK', 'SimHei', 'Microsoft YaHei'])]

if chinese_fonts:
    plt.rcParams['font.sans-serif'] = chinese_fonts[:1]
    print(f"using font: {chinese_fonts[0]}")
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    print("No Chinese font found, using default font")

plt.rcParams['axes.unicode_minus'] = False

def plot_action_distribution(train_paths, test_paths, save_dir, plot_name='data_distribution'):    
    def get_action_distribution(paths):
        action_counts = {}
        for path in paths:
            # Extract action name from path
            # Path format: /aiot-nvme-15T-x2-hk01/siyang/CUHK-X-Final/SM_data/Skeleton/action/user/trial/predictions/sample
            parts = path.split('/')
            action_idx = parts.index('Skeleton') + 1
            action = parts[action_idx]
            action_counts[action] = action_counts.get(action, 0) + 1
        return action_counts

    train_action_dist = get_action_distribution(train_paths)
    test_action_dist = get_action_distribution(test_paths)

    # Sort by total number of samples
    all_actions = sorted(
        set(train_action_dist.keys()) | set(test_action_dist.keys()),
        key=lambda a: train_action_dist.get(a, 0) + test_action_dist.get(a, 0),
        reverse=True
    )

    train_counts = [train_action_dist.get(action, 0) for action in all_actions]
    test_counts = [test_action_dist.get(action, 0) for action in all_actions]

    fig, ax = plt.subplots(figsize=(15, 8))
    x = range(len(all_actions))
    width = 0.35

    ax.bar([i - width/2 for i in x], train_counts, width, label='Train', alpha=0.8)
    ax.bar([i + width/2 for i in x], test_counts, width, label='Test', alpha=0.8)

    ax.set_xlabel('Action')
    ax.set_ylabel('Number of Samples')
    ax.set_title(f'{plot_name.replace("_", " ").title()} Data Distribution by Action')
    ax.set_xticks(x)
    ax.set_xticklabels(all_actions, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{plot_name}.png'), dpi=300, bbox_inches='tight')

    print(f"\n{plot_name.replace('_', ' ').title()} - Train set distribution:")
    for action in all_actions:
        print(f"{action}: {train_action_dist.get(action, 0)}")

    print(f"\n{plot_name.replace('_', ' ').title()} - Test set distribution:")
    for action in all_actions:
        print(f"{action}: {test_action_dist.get(action, 0)}")

    print(f"\nTotal train samples: {sum(train_counts)}")
    print(f"Total test samples: {sum(test_counts)}")
    
    return train_action_dist, test_action_dist


# ########################## cross-trial (80/20 split with stratification)
train_dir = os.path.join(save_dir, 'cross_trial_train.txt')
test_dir = os.path.join(save_dir, 'cross_trial_test.txt')

# Collect all sample paths with their action labels
all_paths = []
all_labels = []

for action in os.listdir(root_dir):
    action_path = os.path.join(root_dir, action)
    if not os.path.isdir(action_path):
        continue
    
    for user in os.listdir(action_path):
        user_path = os.path.join(action_path, user)
        if not os.path.isdir(user_path):
            continue
            
        for trial in os.listdir(user_path):
            trial_path = os.path.join(user_path, trial)
            if not os.path.isdir(trial_path):
                continue
                
            samples_path = os.path.join(trial_path, 'predictions')
            if not os.path.exists(samples_path):
                print(f"Warning: {samples_path} does not exist.")
                continue
                
            for sample in os.listdir(samples_path):
                sample_path = os.path.join(samples_path, sample)
                all_paths.append(sample_path)
                all_labels.append(action)  # Use action as label for stratification

# Perform stratified 80/20 split
train_paths, test_paths = train_test_split(
    all_paths, 
    test_size=0.2, 
    random_state=42, 
    stratify=all_labels
)

with open(train_dir, 'w') as f:
    for path in train_paths:
        f.write(path + '\n')

with open(test_dir, 'w') as f:
    for path in test_paths:
        f.write(path + '\n')

print("Cross-trial split:")
print(f"Train samples: {len(train_paths)}")
print(f"Test samples: {len(test_paths)}")
print(f"Train ratio: {len(train_paths)/(len(train_paths)+len(test_paths))*100:.1f}%")

plot_action_distribution(train_paths, test_paths, save_dir, 'cross_trial_distribution')



#################### cross-subject (with user data quality analysis)

def analyze_user_data_quality(root_dir):
    """
    Analyze each user's data quality:
    - Number of actions covered
    - Total sample count
    - Sample distribution balance (std of samples per action)
    Returns users ranked by data quality (action coverage first, then sample count)
    """
    user_stats = {}
    
    for action in os.listdir(root_dir):
        action_path = os.path.join(root_dir, action)
        if not os.path.isdir(action_path):
            continue
        for user in os.listdir(action_path):
            user_path = os.path.join(action_path, user)
            if not os.path.isdir(user_path):
                continue
            
            if user not in user_stats:
                user_stats[user] = {'actions': {}, 'total_samples': 0}
            
            # Count samples for this user-action pair
            action_samples = 0
            for trial in os.listdir(user_path):
                trial_path = os.path.join(user_path, trial)
                if not os.path.isdir(trial_path):
                    continue
                samples_path = os.path.join(trial_path, 'predictions')
                if os.path.exists(samples_path):
                    action_samples += len([f for f in os.listdir(samples_path) if os.path.isfile(os.path.join(samples_path, f))])
            
            if action_samples > 0:
                user_stats[user]['actions'][action] = action_samples
                user_stats[user]['total_samples'] += action_samples
    
    # Calculate quality metrics for each user
    user_quality = []
    for user, stats in user_stats.items():
        num_actions = len(stats['actions'])
        total_samples = stats['total_samples']
        
        # Calculate balance score (lower std = more balanced)
        if num_actions > 0:
            samples_per_action = list(stats['actions'].values())
            
            balance_score = -np.std(samples_per_action)  # Negative so higher is better
        else:
            balance_score = float('-inf')
        
        user_quality.append({
            'user': user,
            'num_actions': num_actions,
            'total_samples': total_samples,
            'balance_score': balance_score,
            'actions': stats['actions']
        })
    
    # Sort by: 1) num_actions (desc), 2) total_samples (desc), 3) balance_score (desc)
    user_quality.sort(key=lambda x: (x['num_actions'], x['total_samples'], x['balance_score']), reverse=True)
    
    return user_quality

# Analyze user data quality
print("\n" + "="*60)
print("Analyzing user data quality for cross-subject split...")
print("="*60)
user_quality = analyze_user_data_quality(root_dir)

# Print user ranking
print(f"\n{'Rank':<6}{'User':<10}{'Actions':<10}{'Samples':<12}{'Top 3 Actions (samples)'}")
print("-" * 70)
for i, uq in enumerate(user_quality, 1):
    top_actions = sorted(uq['actions'].items(), key=lambda x: x[1], reverse=True)[:3]
    top_str = ', '.join([f"{a}({c})" for a, c in top_actions])
    print(f"{i:<6}{uq['user']:<10}{uq['num_actions']:<10}{uq['total_samples']:<12}{top_str}")

# Select test user: you can choose by rank or by user number
# Option 1: Select by rank (e.g., rank 1 = best data quality)
# test_user_rank = 1  # Select the user with best data quality as test user
# test_user = user_quality[test_user_rank - 1]['user']

# Option 2: Select by user number (traditional method)
test_num = 30  # Change this to select different user
test_user = f'user{test_num}'

# Option 3: Select user with median data quality (balanced choice)
# median_rank = len(user_quality) // 2
# test_user = user_quality[median_rank]['user']

print(f"\nSelected test user: {test_user}")
test_user_info = next((uq for uq in user_quality if uq['user'] == test_user), None)
if test_user_info:
    rank = user_quality.index(test_user_info) + 1
    print(f"  Rank: {rank}/{len(user_quality)}")
    print(f"  Actions covered: {test_user_info['num_actions']}")
    print(f"  Total samples: {test_user_info['total_samples']}")

train_dir = os.path.join(save_dir, f'cross_subject_train_{test_num}.txt')
test_dir = os.path.join(save_dir, f'cross_subject_test_{test_num}.txt')
train_paths = []
test_paths = []
train_subjects = {uq['user'] for uq in user_quality if uq['user'] != test_user}
test_subjects = test_user
for action in os.listdir(root_dir):
    action_path = os.path.join(root_dir, action)
    for user in os.listdir(action_path):
        user_path = os.path.join(action_path, user)
        if user in train_subjects:
            for trial in os.listdir(user_path):
                trial_path = os.path.join(user_path, trial)
                samples_path = os.path.join(trial_path, 'predictions')
                if not os.path.exists(samples_path):
                    print(f"Warning: {samples_path} does not exist.")
                    continue
                for sample in os.listdir(samples_path):
                    sample_path = os.path.join(samples_path, sample)
                    train_paths.append(sample_path)
        elif user in test_subjects:
            for trial in os.listdir(user_path):
                trial_path = os.path.join(user_path, trial)
                samples_path = os.path.join(trial_path, 'predictions')
                if not os.path.exists(samples_path):
                    print(f"Warning: {samples_path} does not exist.")
                    continue
                for sample in os.listdir(samples_path):
                    sample_path = os.path.join(samples_path, sample)
                    test_paths.append(sample_path)
        else:
            print(f"Warning: {user} not in train or test subjects.")
            continue

with open(train_dir, 'w') as f:
    for path in train_paths:
        f.write(path + '\n')

with open(test_dir, 'w') as f:
    for path in test_paths:
        f.write(path + '\n')

print(f"Train samples: {len(train_paths)}")
print(f"Test samples: {len(test_paths)}")

plot_action_distribution(train_paths, test_paths, save_dir, f'cross_subject_distribution_{test_num}')


#################### cross-subject with top-K resampled users (quality-based selection)
# Select top K users by data quality, then perform leave-one-out cross-subject split

resample_users = 20  # Select top 20 users by data quality

# Get top K users from quality ranking (user_quality is already sorted)
top_k_users = [uq['user'] for uq in user_quality[:resample_users]]

print("\n" + "="*60)
print(f"Cross-subject split with TOP {resample_users} resampled users")
print("="*60)
print(f"Selected users (by quality rank): {top_k_users}")

# Show selected users' stats
print(f"\n{'Rank':<6}{'User':<10}{'Actions':<10}{'Samples':<12}")
print("-" * 40)
for i, uq in enumerate(user_quality[:resample_users], 1):
    print(f"{i:<6}{uq['user']:<10}{uq['num_actions']:<10}{uq['total_samples']:<12}")

# Select which user from top-K to use as test user
# Option 1: By rank within top-K (e.g., 1 = best quality user in top-K)
test_rank_in_topk = 1  # Change this: 1-20 for top 20 users
test_user_topk = top_k_users[test_rank_in_topk - 1]

# Option 2: By specific user name
# test_user_topk = 'user5'  # Must be in top_k_users list

print(f"\nTest user: {test_user_topk} (rank {test_rank_in_topk} in top-{resample_users})")

# Build train/test sets using only top-K users
train_subjects_topk = set(top_k_users) - {test_user_topk}
test_subjects_topk = test_user_topk

train_paths_topk = []
test_paths_topk = []

for action in os.listdir(root_dir):
    action_path = os.path.join(root_dir, action)
    if not os.path.isdir(action_path):
        continue
    for user in os.listdir(action_path):
        user_path = os.path.join(action_path, user)
        if not os.path.isdir(user_path):
            continue
        
        # Only process users in top-K list
        if user not in set(top_k_users):
            continue
            
        for trial in os.listdir(user_path):
            trial_path = os.path.join(user_path, trial)
            if not os.path.isdir(trial_path):
                continue
            samples_path = os.path.join(trial_path, 'predictions')
            if not os.path.exists(samples_path):
                continue
            for sample in os.listdir(samples_path):
                sample_path = os.path.join(samples_path, sample)
                if user in train_subjects_topk:
                    train_paths_topk.append(sample_path)
                elif user == test_subjects_topk:
                    test_paths_topk.append(sample_path)

# Save to files
topk_tag = f"top{resample_users}_test{test_rank_in_topk}"
train_dir_topk = os.path.join(save_dir, f'cross_subject_train_{topk_tag}.txt')
test_dir_topk = os.path.join(save_dir, f'cross_subject_test_{topk_tag}.txt')

with open(train_dir_topk, 'w') as f:
    for path in train_paths_topk:
        f.write(path + '\n')

with open(test_dir_topk, 'w') as f:
    for path in test_paths_topk:
        f.write(path + '\n')

print(f"\nTop-{resample_users} Cross-subject split results:")
print(f"  Train users: {len(train_subjects_topk)} users")
print(f"  Test user: {test_user_topk}")
print(f"  Train samples: {len(train_paths_topk)}")
print(f"  Test samples: {len(test_paths_topk)}")
print(f"  Output files: {topk_tag}")

plot_action_distribution(train_paths_topk, test_paths_topk, save_dir, f'cross_subject_distribution_{topk_tag}')


#################### cross-subject-trial (choose specific users, then perform 80/20 random split)
# You can specify which users' data to use for training and testing
# Examples:
# selected_users = list(range(1, 31))     # Use all 30 users (default)
# selected_users = list(range(1, 21))     # Use only the first 20 users (user1-user20)
# selected_users = [1,2,3,5,10,15,20,25]  # Use only specific users
# selected_users = list(range(1, 31))     # Exclude certain users: first select all, then exclude with list comprehension
# selected_users = [i for i in range(1, 31) if i not in [5, 10, 15]]  # Exclude user5, user10, user15

selected_users = list(range(11, 31)) 

if selected_users == list(range(11, 31)):
    user_range_tag = "20users"  
elif len(selected_users) == len(range(min(selected_users), max(selected_users)+1)):
    user_range_tag = f"user{min(selected_users)}-{max(selected_users)}"
else:
    user_range_tag = f"{len(selected_users)}users"

train_dir = os.path.join(save_dir, f'cross_subject_trial_train_{user_range_tag}.txt')
test_dir = os.path.join(save_dir, f'cross_subject_trial_test_{user_range_tag}.txt')

# Collect data only from selected users
selected_paths = []
selected_labels = []
selected_user_set = {f'user{i}' for i in selected_users}

for action in os.listdir(root_dir):
    action_path = os.path.join(root_dir, action)
    if not os.path.isdir(action_path):
        continue
    
    for user in os.listdir(action_path):
        if user not in selected_user_set:
            continue  # Skip users not in selected list
            
        user_path = os.path.join(action_path, user)
        if not os.path.isdir(user_path):
            continue
            
        for trial in os.listdir(user_path):
            trial_path = os.path.join(user_path, trial)
            if not os.path.isdir(trial_path):
                continue
                
            samples_path = os.path.join(trial_path, 'predictions')
            if not os.path.exists(samples_path):
                print(f"Warning: {samples_path} does not exist.")
                continue
                
            for sample in os.listdir(samples_path):
                sample_path = os.path.join(samples_path, sample)
                selected_paths.append(sample_path)
                selected_labels.append(action)

# Perform stratified 80/20 split on selected users' data
train_paths, test_paths = train_test_split(
    selected_paths, 
    test_size=0.2, 
    random_state=42, 
    stratify=selected_labels
)

with open(train_dir, 'w') as f:
    for path in train_paths:
        f.write(path + '\n')

with open(test_dir, 'w') as f:
    for path in test_paths:
        f.write(path + '\n')

print(f"\nCross-subject-trial split (using {len(selected_users)} users: {user_range_tag}):")
print(f"Train samples: {len(train_paths)}")
print(f"Test samples: {len(test_paths)}")
print(f"Train ratio: {len(train_paths)/(len(train_paths)+len(test_paths))*100:.1f}%")

plot_action_distribution(train_paths, test_paths, save_dir, f'cross_subject_trial_{user_range_tag}_distribution')


# cross-domain