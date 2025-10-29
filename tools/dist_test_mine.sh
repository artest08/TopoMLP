#!/usr/bin/env bash

CONFIG=projects/configs/topomlp_setA_r50_wo_yolov8.py
CHECKPOINT=work_dirs/topomlp_setA_r50_wo_yolov8/topomlp_setA_r50_wo_yolov8_e24.pth

GPUS=$1
PORT=${PORT:-29500}

WORK_DIR=work_dirs/toponet_mlp_sA_test
timestamp=`date +"%y%m%d.%H%M%S"`

mkdir -p ${WORK_DIR}

~/containers/python_topomlp -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    --eval openlane_v2 ${@:2} \
    2>&1 | tee ${WORK_DIR}/test.${timestamp}.log
