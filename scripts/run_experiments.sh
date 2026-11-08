#!/usr/bin/env bash
# Task 3.5 - six+ runs, each differing from a neighbour by ONE meaningful change.
set -e
python -m vg.train --model baseline    --run-name r1_baseline_cnn          --epochs 25 --ft-epochs 0
python -m vg.train --model transfer    --run-name r2_mnv2_head_only        --no-finetune
python -m vg.train --model transfer    --run-name r3_mnv2_finetune         # change vs r2: unfreeze last block
python -m vg.train --model transfer    --run-name r4_mnv2_ft_noaug         --no-augment   # change vs r3: no augmentation
python -m vg.train --model transfer    --run-name r5_mnv2_ft_lr3e-4        --lr 3e-4      # change vs r3: head learning rate
python -m vg.train --model transfer    --run-name r6_mnv2_ft_img160        --img-size 160 # change vs r3: image size
python -m vg.train --model transfer    --backbone efficientnetb0 --run-name r7_effb0_finetune  # change vs r3: backbone
# 3.3 manual-vs-autolog pair (same settings as r3):
python -m vg.train --model transfer    --run-name r8_mnv2_ft_autolog       --tracking autolog
