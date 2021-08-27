#!/bin/bash 
####### not working need tweaking
EXP_NAME="SwedishGPT3_2.7B_OriginalMegatron"
 # ngc args
INSTANCE="dgx1v.32g.8.norm"
IMAGE="nvcr.io/nvidia/pytorch:20.11-py3"
# wandb args
PROJECT_NAME=SwedishGPT3_2.7B_OriginalMegatron
# megatron-lm args
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
DATA_PATH=/raid/SV_CC100Sprakbank_text_document
CHECKPOINT_PATH=/result
VOCAB_FILE=/mnt/dataset/32k/vocab.json
MERGE_FILE=/mnt/dataset/32k/merges.txt

MP_SIZE=8
DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} --nnodes ${NNODES} --node_rank ${NODE_RANK} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"
GPT_ARGS="--num-layers 32 \
           --hidden-size 2560 \
           --num-attention-heads 32 \
           --seq-length 512 \
           --max-position-embeddings 1024 \
           --lr 0.00015 \
           --train-iters 5000000 \
           --min-lr 0.00001 \
           --lr-decay-iters 990000 \
           --lr-warmup-fraction 0.01 \
           --override-lr-scheduler \
           --micro-batch-size 2 \
           --vocab-file ${VOCAB_FILE} \
           --merge-file ${MERGE_FILE} \
           --split 949,50,1 \
           --distributed-backend nccl \
           --fp16"

OUTPUT_ARGS="--log-interval 10000 \
             --save-interval 500000 \
             --eval-interval 500000 \
             --eval-iters 100000 \
             --checkpoint-activations"
CMD="python -m torch.distributed.launch ${DISTRIBUTED_ARGS} \
    pretrain_gpt.py \
        --tensor-model-parallel-size 2 \
        --pipeline-model-parallel-size 2 \
        ${GPT_ARGS} \
        ${OUTPUT_ARGS} \
        --save ${CHECKPOINT_PATH} \
        --load ${CHECKPOINT_PATH} \
        --data-path ${DATA_PATH}
        --tensorboard-dir ${CHECKPOINT_PATH} "
echo "${CMD}"
ngc batch run \
--name ${EXP_NAME} --preempt RUNONCE --ace nv-us-west-2 \
--instance ${INSTANCE} \
--commandline "nvidia-smi && \
cp -r /mnt/dataset/32k /raid && \
cp /mnt/dataset/SV_CC100Sprakbank_text_document.bin /raid/ && \
cp /mnt/dataset/SV_CC100Sprakbank_text_document.idx /raid/ && \
cp -r /mnt/ckpt/iter_1000000 /result && \
cp /mnt/ckpt/latest_checkpointed_iteration.txt /result && \
ls /raid && \
git clone https://github.com/NVIDIA/Megatron-LM.git && \
cd Megatron-LM/ && \
${CMD}" \
--result /result \
--image ${IMAGE} \
--org nvidian \
--datasetid 80889:/mnt/dataset \
--datasetid 84035:/mnt/ckpt \
--port 6006