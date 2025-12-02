import torch
import numpy as np
import os
import random
import copy
from torch.utils.data import Dataset, DataLoader
from utils.utils_data import crop_scale, resample
from utils.tools import read_pkl
import re
import json
import pandas as pd
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
from collections import Counter


actions = {
    'ElbowPunch': 0,
    'FrontKick': 1,
    'FrontPunch': 2,
    'HighKick': 3,
    'HookPunch': 4,
    'JumpingJack': 5,
    'KneeKick': 6,
    'LegBack': 7,
    'LegCross': 8,
    'RonddeJambe': 9,
    'Running': 10,
    'Shuffle': 11,
    'SideLunges': 12,
    'SlowSkater': 13,
    'Squat': 14
}

def joint_mapping(source_format, target_format):
    mapping = np.ones(len(target_format),dtype=np.int32)*-1
    for joint_name in target_format:
        if joint_name in source_format:
            mapping[target_format[joint_name]] = source_format[joint_name]
    return np.array(mapping)

OpenPose_25 = {
    'Nose':0, 'Neck':1, 'R_Shoulder':2, 'R_Elbow':3, 'R_Wrist':4, 'L_Shoulder':5, 'L_Elbow':6, \
    'L_Wrist':7, 'Pelvis':8, 'R_Hip': 9, 'R_Knee':10, 'R_Ankle':11, 'L_Hip':12, 'L_Knee':13, 'L_Ankle':14, \
    'R_Eye':15, 'L_Eye':16, 'R_Ear':17, 'L_Ear':18, 'L_Toe':19, 'L_Foot':20, 'L_Heel':21, 'R_Toe':22, 'R_Foot':23, 'R_Heel':24
    }

SMPL_22 = {
    'Pelvis':0, 'L_Hip':1, 'R_Hip':2, 'Spine': 3, 'L_Knee':4, 'R_Knee':5, 'Thorax': 6, 'L_Ankle':7, 'R_Ankle':8,'Thorax_up':9, \
    'L_Foot':10, 'R_Foot':11, 'Neck': 12, 'L_Collar':13, 'R_Collar':14, 'Jaw':15, 'L_Shoulder':16, 'R_Shoulder':17,\
    'L_Elbow':18, 'R_Elbow':19, 'L_Wrist': 20, 'R_Wrist': 21}

OpenPose_18 = {
    'Nose': 0,  'Neck': 1,  'R_Shoulder': 2,  'R_Elbow': 3,  'R_Wrist': 4, 'L_Shoulder': 5, 
    'L_Elbow': 6, 'L_Wrist': 7, 'R_Hip': 8, 'R_Knee': 9, 
    'R_Ankle': 10, 'L_Hip': 11, 'L_Knee': 12, 'L_Ankle': 13, 'R_Eye': 14, 'L_Eye': 15, 'R_Ear': 16, 'L_Ear': 17
}

H36M_17 = {
    'root': 0,
    'rhip': 1,
    'rkne': 2,
    'rank': 3,
    'lhip': 4,
    'lkne': 5,
    'lank': 6,
    'belly': 7,
    'neck': 8,
    'nose': 9,
    'head': 10,
    'lsho': 11,
    'lelb': 12,
    'lwri': 13,
    'rsho': 14,
    'relb': 15,
    'rwri': 16
}

def openpose25to18(op_joints25):
    mapping = joint_mapping(OpenPose_25, OpenPose_18)
    N = op_joints25.shape[0] 
    op_joints18 = np.ones((N, 18, 3)) * np.nan   
    for idx, map_idx in enumerate(mapping):
        if map_idx != -1:
            op_joints18[:, idx, :] = op_joints25[:, map_idx, :]
    return op_joints18



def smpl2openpose(smpl_joints):
    N = smpl_joints.shape[0]
    # Create an array to hold the transformed joints for OpenPose
    openpose_joints = np.ones((N, 18, 3)) * np.nan  # Initialize with NaNs for unmatched joints

    # Get the mapping from SMPL to OpenPose
    mapping = joint_mapping(SMPL_22, OpenPose_18)
    
    # Apply the mapping to transform coordinates
    for idx, map_idx in enumerate(mapping):
        if map_idx != -1:
            openpose_joints[:, idx, :] = smpl_joints[:, map_idx, :]
    return openpose_joints

def get_action_names(file_path = "data/action/ntu_actions.txt"):
    f = open(file_path, "r")
    s = f.read()
    actions = s.split('\n')
    action_names = []
    for a in actions:
        action_names.append(a.split('.')[1][1:])
    return action_names

def make_cam(x, img_shape):
    '''
        Input: x (M x T x V x C)
               img_shape (height, width)
    '''
    h, w = img_shape
    if w >= h:
        x_cam = x / w * 2 - 1
    else:
        x_cam = x / h * 2 - 1
    return x_cam

def coco2h36m(x):
    '''
        Input: x (M x T x V x C)
        
        COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
        
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = np.zeros(x.shape)
    y[:,:,0,:] = (x[:,:,11,:] + x[:,:,12,:]) * 0.5
    y[:,:,1,:] = x[:,:,12,:]
    y[:,:,2,:] = x[:,:,14,:]
    y[:,:,3,:] = x[:,:,16,:]
    y[:,:,4,:] = x[:,:,11,:]
    y[:,:,5,:] = x[:,:,13,:]
    y[:,:,6,:] = x[:,:,15,:]
    y[:,:,8,:] = (x[:,:,5,:] + x[:,:,6,:]) * 0.5
    y[:,:,7,:] = (y[:,:,0,:] + y[:,:,8,:]) * 0.5
    y[:,:,9,:] = x[:,:,0,:]
    y[:,:,10,:] = (x[:,:,1,:] + x[:,:,2,:]) * 0.5
    y[:,:,11,:] = x[:,:,5,:]
    y[:,:,12,:] = x[:,:,7,:]
    y[:,:,13,:] = x[:,:,9,:]
    y[:,:,14,:] = x[:,:,6,:]
    y[:,:,15,:] = x[:,:,8,:]
    y[:,:,16,:] = x[:,:,10,:]
    return y

def smpljoints2h36m(x):
    '''Input: x (M x T x V x C)
    M: number of persons; 
    T: number of frames (same as total_frames); 
    V: number of keypoints; 
    C: number of dimensions for keypoint coordinates (C=2 for 2D keypoint, C=3 for 3D keypoint)
    '''
    h36m_joints = np.zeros((x.shape[0], x.shape[1], 17, 3))  
    mapping = {
        0: 0,  # root -> Pelvis
        1: 2,  # rhip -> Right Hip
        2: 5,  # rkne -> Right Knee
        3: 8,  # rank -> Right Ankle
        4: 1,  # lhip -> Left Hip
        5: 4,  # lkne -> Left Knee
        6: 7,  # lank -> Left Ankle
        7: 3,  # belly -> Spine1
        8: 12, # neck -> Neck
        10: 15, # head -> Head
        11: 16, # lsho -> Left Shoulder
        12: 18, # lelb -> Left Elbow
        13: 20, # lwri -> Left Wrist
        14: 17, # rsho -> Right Shoulder
        15: 19, # relb -> Right Elbow
        16: 21  # rwri -> Right Wrist
    }

    for h36m_idx, smpl_idx in mapping.items():
        h36m_joints[:,:,h36m_idx,:] = x[:,:,smpl_idx,:]

    h36m_joints[:,:,9,:] = (x[:,:,12,:] + x[:,:,15,:]) / 2
    return h36m_joints

def random_move(data_numpy,
                angle_range=[-10., 10.],
                scale_range=[0.9, 1.1],
                transform_range=[-0.1, 0.1],
                move_time_candidate=[1]):
    data_numpy = np.transpose(data_numpy, (3,1,2,0)) # M,T,V,C-> C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)
    A = np.random.uniform(angle_range[0], angle_range[1], num_node)
    S = np.random.uniform(scale_range[0], scale_range[1], num_node)
    T_x = np.random.uniform(transform_range[0], transform_range[1], num_node)
    T_y = np.random.uniform(transform_range[0], transform_range[1], num_node)
    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)
    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1], node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1], node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1], node[i + 1] - node[i])
    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])
    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)
    data_numpy = np.transpose(data_numpy, (3,1,2,0)) # C,T,V,M -> M,T,V,C
    return data_numpy    

def create_weighted_sampler(labels):
    class_counts = Counter(labels)
    total_samples = len(labels)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

class cuhkxDataset(Dataset):
    def __init__(self, data_split, n_frames=1, scale_range=[1,1]):
        self.n_frames = n_frames
        self.scale_range = scale_range

        with open(data_split) as f:
            lines = f.readlines()
            self.all_files = lines
            file_count = len(self.all_files)
            print('---- File count:', file_count)
        motions = []  
        labels = []
        for filepath in self.all_files:
            filepath = filepath.strip()
            match = re.search(r'/(\d+)[^\d/]*/', filepath)
            if match:
                action_number = match.group(1)
                label = int(action_number)  # 动作编号已经是0-43，直接使用
            labels.append(label)

            with open(filepath) as f:
                data = json.load(f)[0]
                keypoints_conf = np.array(data['keypoint_scores'])[..., None]
                keypoints_cam = np.array(data['keypoints'])  
                # keypoints = np.concatenate((keypoints_cam, keypoints_conf), axis=-1)
                keypoints = keypoints_cam
            
            # # mmpose format to openpose format
            # keypoints = np.expand_dims(keypoints, axis=0)
            # kpt_thr = 0.3
            # # compute neck joint
            # neck = (keypoints[:, 5] + keypoints[:, 6]) / 2
            # if keypoints[:, 5, 2] < kpt_thr or keypoints[:, 6, 2] < kpt_thr:
            #     neck[:, 2] = 0
            # # 17 keypoints to 18 keypoints
            # new_keypoints = np.insert(keypoints[:, ], 17, neck, axis=1)
            # openpose_idx = [15, 14, 17, 16, 2, 6, 3, 7, 4, 8, 12, 9, 13, 10, 1]
            # mmpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
            # new_keypoints[:, openpose_idx, :] = new_keypoints[:, mmpose_idx, :]
            # threed_joints = np.array(new_keypoints)
            # motion = np.expand_dims(threed_joints, axis=0)  # (1, 1, 18, 3)
            # motions.append(motion)
            
            keypoints = np.expand_dims(keypoints, axis=0)
            threed_joints = np.array(keypoints)
            motion = np.expand_dims(threed_joints, axis=0)  # (1, 1, 17, 3)
            motions.append(motion)

        
        self.motions = np.array(motions)
        self.labels = np.array(labels)
            
    def __getitem__(self, idx):
        'Generates one sample of data'
        motion, label = self.motions[idx], self.labels[idx]
        if self.scale_range:
            result = crop_scale(motion, scale_range=self.scale_range)
        else:
            result = motion
        return result.astype(np.float32), label

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.motions)
