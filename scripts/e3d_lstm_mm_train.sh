# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

cd ..
python run.py \
    --train_data_paths data/milan_tra.npy \
    --valid_data_paths data/milan_val.npy \
    --test_data_paths data/milan_test.npy \
    --save_dir checkpoints/8-2-2/8-8-4 \
    --gen_frm_dir results/8-2-2/8-8-4 \
    --is_training False \
    --dataset_name milan \
    --input_seq_length 8 \
    --output_seq_length 2 \
    --dimension_3D 2 \
    --img_width 100 \
    --model_name e3d_lstm \
    --num_hidden 8,8,4 \
    --filter_size 10 \
    --layer_norm True \
    --scheduled_sampling True \
    --sampling_stop_iter 40 \
    --sampling_start_value 1.0 \
    --sampling_delta_per_iter 0.00002 \
    --lr 0.1 \
    --batch_size 8 \
    --max_iterations 200 \
    --display_interval 1 \
    --test_interval 1 \
    --snapshot_interval 200 \
    --allow_gpu_growth True
