#!/usr/bin/env bash

CONFIG=projects/configs/topomlp_setA_r50_wo_yolov8_near.py
GPUS=$1
PORT=${PORT:-29500}

WORK_DIR=work_dirs/topomlp_sA_near_v2
mkdir -p ${WORK_DIR}
timestamp=`date +"%y%m%d.%H%M%S"`

~/containers/python_topomlp -m torch.distributed.run \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py \
    $CONFIG \
    --seed 42 \
    --work-dir ${WORK_DIR} \
    --launcher pytorch ${@:2} \
    2>&1 | tee ${WORK_DIR}/train.${timestamp}.log
