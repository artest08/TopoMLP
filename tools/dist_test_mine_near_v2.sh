#!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`

WORK_DIR=work_dirs/topomlp_sA_near
CONFIG=projects/configs/topomlp_setA_r50_wo_yolov8_near.py

GPUS=$1
CHECKPOINT_NAME=$2
PORT=${PORT:-28510}

CHECKPOINT=${WORK_DIR}/${CHECKPOINT_NAME}
mkdir -p ${WORK_DIR}/test

~/containers/python_topomlp -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    --eval openlane_v2 ${@:3} \
    2>&1 | tee ${WORK_DIR}/test/test.${CHECKPOINT_NAME}.log
