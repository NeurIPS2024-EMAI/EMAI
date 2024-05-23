#!/bin/sh
seed_max=1

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 \
    python ../../src/main.py --config qmix --env-config sc2 --run masker \
    --env_args.map_name 8m --run_mode Test --runner episode --batch_size_run 1 \
    --checkpoint_path path_of_target_agents_model --masker_checkpoint_path path_of_masking_agents_model \
    --need_attack False --att_size 1 --noise_w 1 \
    --need_patch False --need_collect False --patch_budget 50 --diff_th 2 --ep_reward_th 19.5
done