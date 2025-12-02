# imu data training

```bash
dataset: example data data_imu: all data
 
first step : run the data_reader.ipynb

second step : run the dataset_maker.ipynb

third step : run the data_analysisipynb
```

```bash

##run cross trail
bash ./command_accgyrmag_transformer_crosstrail.sh

log: ./activity_40/cnn_transformer/acc_gyr_mag/bsz128_all

##run cross user 
bash ./command_activity20_accgyrmag_transformer_crossuser.sh
log: ./runs/activity_20/cross_user/cnn_transformer/acc_gyr_mag/bsz128

##run cross user resample
bash ./command_activity20_accgyrmag_transformer_crossuser.sh
log: ./runs/activity_20/cross_user/cnn_transformer/acc_gyr_mag/bsz128_resample_all