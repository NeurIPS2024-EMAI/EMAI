#!/bin/sh
env="MPE"
scenario="simple_reference"
num_landmarks=3
num_agents=2
algo="rmappo"
exp="check"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python ../../src/render_mpe.py --save_gifs --share_policy --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 25 --render_episodes 5 \
    --run_policy masker --model_dir path_of_target_agents_model --masker_model_dir path_of_masking_agents_model agent_selection masker \
    --need_RRD True --need_attack False --need_patch = False --need_collect False --patch_budget 50 --ep_reward_th -25 --diff_th 0.5
done
