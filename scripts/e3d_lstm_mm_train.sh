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
    --input_seq_length 4 \
    --output_seq_length 4 \
    --dimension_3D 1 \
    --num_hidden 4,4 \
    --filter_size 2 \
    --lr 0.001 \
    --batch_size 2 \
    --max_iterations 5 \
    --is_training True \
    --pretrained_model checkpoints/model.ckpt \
    --save_dir checkpoints/ \
    --gen_frm_dir results/ \
    --train_data_paths ../data/milan_tra.npy \
    --valid_data_paths ../data/milan_val.npy \
    --test_data_paths ../data/milan_test.npy \
    --dataset_name milan \
    --img_width 100 \
    --model_name e3d_lstm \
    --layer_norm True \
    --scheduled_sampling True \
    --sampling_stop_iter 40 \
    --sampling_start_value 1.0 \
    --sampling_delta_per_iter 0.00002 \
    --display_interval 1 \
    --test_interval 1 \
    --snapshot_interval 1 \
    --allow_gpu_growth True
