cd ..
python run.py \
    --pretrained_model pretrain/model.ckpt-2350 \
    --input_seq_length 6 \
    --output_seq_length 6 \
    --dimension_3D 3 \
    --num_hidden 25,25 \
    --filter_size 20 \
    --lr 0.001 \
    --batch_size 16 \
    --max_iterations 10000 \
    --is_training True \
    --save_dir checkpoints/6-6/25-25 \
    --gen_frm_dir results/6-6/25-25 \
    --train_data_paths data/milan_tra.npy \
    --valid_data_paths data/milan_val.npy \
    --test_data_paths data/milan_val.npy \
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
