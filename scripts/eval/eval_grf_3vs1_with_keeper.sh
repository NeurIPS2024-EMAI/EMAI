#!/bin/sh
seed_max=1

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 \
    python ../../src/main.py --config=vdn_gfootball --env-config=gfootball --run default \
    --env_args.map_name academy_3_vs_1_with_keeper --run_mode Test --runner episode --batch_size_run 1 \
    --checkpoint_path path_of_target_agents_model --masker_checkpoint_path path_of_masking_agents_model \
    --need_attack  False --noise_w 0.1 --att_size 1 \
    --need_patch False --need_collect False --patch_budget 20 --ep_reward_th 6 --diff_th 0.5
done