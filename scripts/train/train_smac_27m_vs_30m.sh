#!/bin/sh
seed_max=1

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 \
    python ../../src/main.py --config qmix_large --env-config sc2 --run default \
    --env_args.map_name 27m_vs_30m --run_mode Train --runner parallel --batch_size_run 8
done