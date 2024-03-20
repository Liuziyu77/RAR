#!/bin/bash

# custom config
DATA=./data
TRAINER=ZeroshotCLIP_topk
DATASET=fgvc_aircraft #
CFG=vit_b16  # rn50, rn101, vit_b32 or vit_b16

python eval.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/zero-shot/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
