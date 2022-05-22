#!/usr/bin/env bash
###
 # @Author       : Zihao Zhao
 # @E-mail       : zhzhao18@fudan.edu.cn
 # @Company      : IBICAS, Fudan University
 # @Date         : 2022-05-15 23:41:19
 # @LastEditors  : Zihao Zhao
 # @LastEditTime : 2022-05-22 18:53:37
 # @FilePath     : /mmtracking/bash/mstracker_train.sh
 # @Description  : 
### 

exp_name="mstracker_coco_vot2018"
CONFIG="/zhzhao/code/mmtracking_master_20220513/configs/sot/mstracker/mstracker_20e_vot2018_coco.py"
GPUS=2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

/zhzhao/miniconda3/envs/mmdet/bin/python3.7 -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    /zhzhao/code/mmtracking_master_20220513/tools/train.py \
    $CONFIG \
    --work-dir /zhzhao/code/mmtracking_master_20220513/work_dirs/${exp_name} \
    --launcher pytorch \
    ${@:3} \
> /zhzhao/code/mmtracking_master_20220513/sys_log/${exp_name}.txt 2>&1
