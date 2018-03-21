#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
#知道脚本内部执行的变量的值或执行结果
set -e
#就是当命令以非零状态退出时，则退出shell。主要作用是，当脚本执行出现意料之外的情况时，立即退出，避免错误被忽略，导致最终结果不正确

export PYTHONUNBUFFERED="True"

DEV=$1 #gpu
DEV_ID=$2 #0
NET=$3 #VGG16 
DATASET=$4 # pascal_voc

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=70000
    ;;
  coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    ITERS=490000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/faster_rcnn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./tools/train_net.py --device ${DEV} --device_id ${DEV_ID} \
  --weights data/pretrain_model/VGG_imagenet.npy \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  --network VGGnet_train \
  ${EXTRA_ARGS}
  
#time python ./tools/train_net.py --device gpu --device_id 0 --weights data/pretrain_model/VGG_imagenet.npy 
#  --imdb voc_2007_trainval --iters 70000 --cfg experiments/cfgs/faster_rcnn_end2end.yml --network VGGnet_train ${EXTRA_ARGS}

 

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time python ./tools/test_net.py --device ${DEV} --device_id ${DEV_ID} \
  --weights ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  --network VGGnet_test \
  ${EXTRA_ARGS}

