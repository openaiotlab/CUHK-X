#CUDA_VISIBLE_DEVICES=0,1,2,3 python CUHK-X-VLM/src/sequential_action_recording/main_internvl_shuffle.py --modality depth --model_size 2B
#CUDA_VISIBLE_DEVICES=0,1,2,3 python CUHK-X-VLM/src/sequential_action_recording/main_internvl_shuffle.py --modality depth --model_size 8B
#CUDA_VISIBLE_DEVICES=0,1,2,3 python CUHK-X-VLM/src/sequential_action_recording/main_qwenvl_shuffle.py --modality depth --model_size 3B
#CUDA_VISIBLE_DEVICES=0,1,2,3 python CUHK-X-VLM/src/sequential_action_recording/main_qwenvl_shuffle.py --modality depth --model_size 7B
CUDA_VISIBLE_DEVICES=0,1,2,3 python CUHK-X-VLM/src/sequential_action_recording/main_videochatr1_shuffle.py --modality depth
#CUDA_VISIBLE_DEVICES=0,1,2,3 python CUHK-X-VLM/src/sequential_action_recording/main_videollava_shuffle.py --modality depth

#CUDA_VISIBLE_DEVICES=0,1,2,3 python CUHK-X-VLM/src/sequential_action_recording/main_internvl_shuffle.py --modality rgb --model_size 2B
#CUDA_VISIBLE_DEVICES=0,1,2,3 python CUHK-X-VLM/src/sequential_action_recording/main_internvl_shuffle.py --modality rgb --model_size 8B
#CUDA_VISIBLE_DEVICES=0,1,2,3 python CUHK-X-VLM/src/sequential_action_recording/main_qwenvl_shuffle.py --modality rgb --model_size 3B
#CUDA_VISIBLE_DEVICES=0,1,2,3 python CUHK-X-VLM/src/sequential_action_recording/main_qwenvl_shuffle.py --modality rgb --model_size 7B
#CUDA_VISIBLE_DEVICES=0,1,2,3 python CUHK-X-VLM/src/sequential_action_recording/main_videochatr1_shuffle.py --modality rgb
#CUDA_VISIBLE_DEVICES=0,1,2,3 python CUHK-X-VLM/src/sequential_action_recording/main_videollava_shuffle.py --modality rgb
