import os
import numpy as np
import random
from typing import Any, List, Optional
import pandas as pd
import scipy
import scipy.io
import pdb
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from scipy.signal import resample


class IMU_DataLoader(Dataset):
    def __init__(self,
                 datapth,
                 data_type,
                 modalities: List[str],
                 sample_length,
                 ):
        '''
        datapth: the path of data csv file
        data_type: string to identify train mode ['pretrain','test', 'supervised']
        modalities : list of modalities [imu, skt] or [imu] or [skt]
        sample_length: the length of the sample
        '''
        super().__init__()
        self.data_type=data_type
        self.modalities = modalities
        self.sample_length=sample_length
        
        self.dataset=pd.read_csv(datapth)
        self.data=self.dataset[self.dataset['data_type']==data_type]
        self.sample_ls=sorted(sorted(list(self.data.index)))
        # self.sample_num=len(self.sample_ls)
        self.subjet_ls=sorted(list(set(self.data['user_id_num'])))
        self.activity_ls=sorted(list(set(self.data['activity_id_num'])))
        
    def get_data_for_instance(self,instance,modality):
        up_data=pd.read_csv(instance['up_path'].iloc[0])
        down_data=pd.read_csv(instance['down_path'].iloc[0])
        wtla_data=up_data[up_data['设备名称'].str.contains('WTLA', na=False)]
        wtc_data=up_data[up_data['设备名称'].str.contains('WTC', na=False)]
        wtra_data=up_data[up_data['设备名称'].str.contains('WTRA', na=False)]
        wtrl_data=down_data[down_data['设备名称'].str.contains('WTRL', na=False)]
        wtll_data=down_data[down_data['设备名称'].str.contains('WTLL', na=False)]
        if modality=='acc':
            wtla_data_acc=wtla_data[['加速度X(g)','加速度Y(g)','加速度Z(g)']].to_numpy()
            wtla_data_acc=resample(wtla_data_acc,self.sample_length) # (sample_length,3)
            wtc_data_acc=wtc_data[['加速度X(g)','加速度Y(g)','加速度Z(g)']].to_numpy()
            wtc_data_acc=resample(wtc_data_acc,self.sample_length)
            wtra_data_acc=wtra_data[['加速度X(g)','加速度Y(g)','加速度Z(g)']].to_numpy()
            wtra_data_acc=resample(wtra_data_acc,self.sample_length)
            wtrl_data_acc=wtrl_data[['加速度X(g)','加速度Y(g)','加速度Z(g)']].to_numpy()
            wtrl_data_acc=resample(wtrl_data_acc,self.sample_length)
            wtll_data_acc=wtll_data[['加速度X(g)','加速度Y(g)','加速度Z(g)']].to_numpy()
            wtll_data_acc=resample(wtll_data_acc,self.sample_length)
            
            # merge the data to be [sample_length,15]
            data=np.concatenate([wtla_data_acc,wtc_data_acc,wtra_data_acc,wtrl_data_acc,wtll_data_acc],axis=1)
            data = data.astype(np.float32)
            data = data.transpose(1, 0)
        elif modality=='gyr':
            wtla_data_gyr=wtla_data[['角速度X(°/s)','角速度Y(°/s)','角速度Z(°/s)']].to_numpy()
            wtla_data_gyr=resample(wtla_data_gyr,self.sample_length)
            wtc_data_gyr=wtc_data[['角速度X(°/s)','角速度Y(°/s)','角速度Z(°/s)']].to_numpy()
            wtc_data_gyr=resample(wtc_data_gyr,self.sample_length)
            wtra_data_gyr=wtra_data[['角速度X(°/s)','角速度Y(°/s)','角速度Z(°/s)']].to_numpy()
            wtra_data_gyr=resample(wtra_data_gyr,self.sample_length)
            wtrl_data_gyr=wtrl_data[['角速度X(°/s)','角速度Y(°/s)','角速度Z(°/s)']].to_numpy()
            wtrl_data_gyr=resample(wtrl_data_gyr,self.sample_length)
            wtll_data_gyr=wtll_data[['角速度X(°/s)','角速度Y(°/s)','角速度Z(°/s)']].to_numpy()
            wtll_data_gyr=resample(wtll_data_gyr,self.sample_length)
            
            # merge the data to be [sample_length,15]
            data=np.concatenate([wtla_data_gyr,wtc_data_gyr,wtra_data_gyr,wtrl_data_gyr,wtll_data_gyr],axis=1)
            data = data.astype(np.float32)
            data = data.transpose(1, 0)
        elif modality=='mag':
            wtla_data_mag=wtla_data[['磁场X(uT)','磁场Y(uT)','磁场Z(uT)']].to_numpy()
            wtla_data_mag=resample(wtla_data_mag,self.sample_length)
            wtc_data_mag=wtc_data[['磁场X(uT)','磁场Y(uT)','磁场Z(uT)']].to_numpy()
            wtc_data_mag=resample(wtc_data_mag,self.sample_length)
            wtra_data_mag=wtra_data[['磁场X(uT)','磁场Y(uT)','磁场Z(uT)']].to_numpy()
            wtra_data_mag=resample(wtra_data_mag,self.sample_length)
            wtrl_data_mag=wtrl_data[['磁场X(uT)','磁场Y(uT)','磁场Z(uT)']].to_numpy()
            wtrl_data_mag=resample(wtrl_data_mag,self.sample_length)
            wtll_data_mag=wtll_data[['磁场X(uT)','磁场Y(uT)','磁场Z(uT)']].to_numpy()
            wtll_data_mag=resample(wtll_data_mag,self.sample_length)
            
            # merge the data to be [sample_length,15]
            data=np.concatenate([wtla_data_mag,wtc_data_mag,wtra_data_mag,wtrl_data_mag,wtll_data_mag],axis=1)
            data = data.astype(np.float32)
            data = data.transpose(1, 0)
        elif modality=='acc_gyr_mag':
            wtla_data_acc_gyr_mag=wtla_data[['加速度X(g)','加速度Y(g)','加速度Z(g)','角速度X(°/s)','角速度Y(°/s)','角速度Z(°/s)','磁场X(uT)','磁场Y(uT)','磁场Z(uT)']].to_numpy()
            wtla_data_acc_gyr_mag=resample(wtla_data_acc_gyr_mag,self.sample_length)
            wtc_data_acc_gyr_mag=wtc_data[['加速度X(g)','加速度Y(g)','加速度Z(g)','角速度X(°/s)','角速度Y(°/s)','角速度Z(°/s)','磁场X(uT)','磁场Y(uT)','磁场Z(uT)']].to_numpy()
            wtc_data_acc_gyr_mag=resample(wtc_data_acc_gyr_mag,self.sample_length)
            wtra_data_acc_gyr_mag=wtra_data[['加速度X(g)','加速度Y(g)','加速度Z(g)','角速度X(°/s)','角速度Y(°/s)','角速度Z(°/s)','磁场X(uT)','磁场Y(uT)','磁场Z(uT)']].to_numpy()
            wtra_data_acc_gyr_mag=resample(wtra_data_acc_gyr_mag,self.sample_length)
            wtrl_data_acc_gyr_mag=wtrl_data[['加速度X(g)','加速度Y(g)','加速度Z(g)','角速度X(°/s)','角速度Y(°/s)','角速度Z(°/s)','磁场X(uT)','磁场Y(uT)','磁场Z(uT)']].to_numpy()
            wtrl_data_acc_gyr_mag=resample(wtrl_data_acc_gyr_mag,self.sample_length)
            wtll_data_acc_gyr_mag=wtll_data[['加速度X(g)','加速度Y(g)','加速度Z(g)','角速度X(°/s)','角速度Y(°/s)','角速度Z(°/s)','磁场X(uT)','磁场Y(uT)','磁场Z(uT)']].to_numpy()
            wtll_data_acc_gyr_mag=resample(wtll_data_acc_gyr_mag,self.sample_length)
            
            # merge the data to be [sample_length,15]
            data=np.concatenate([wtla_data_acc_gyr_mag,wtc_data_acc_gyr_mag,wtra_data_acc_gyr_mag,wtrl_data_acc_gyr_mag,wtll_data_acc_gyr_mag],axis=1)
            data = data.astype(np.float32)
            data = data.transpose(1, 0)
        
        return data
        
    def __len__(self):
        return len(self.sample_ls)
    
    def __getitem__(self, index):
        data_sample={}
        instance=self.data[self.data.index == self.sample_ls[index]] # a row
        # label is the index of activity_id_num in self.activity_ls
        
        # label=self.activity_ls.index(instance['activity_id_num'].unique()[0])
        label=instance['activity_id_num'].unique()[0]

        data_sample["label"] = label
        # data
        # for modality in self.modalities:
        modality=self.modalities
        data=self.get_data_for_instance(instance,modality)

        data_sample[modality] = data
        return data_sample


if __name__ == '__main__':
    datapth='../dataset/activity_11_12_13_22_9/dataset_user_1.csv'
    modalities='acc_gyr_mag'
    sample_length=50
    dataloader=IMU_DataLoader(datapth,data_type='train',modalities=modalities,sample_length=sample_length)
    data=dataloader.__getitem__(0) # a dict with key 'acc' and value is a numpy array of shape (sample_length,15)
    print(data['acc_gyr_mag'].shape)