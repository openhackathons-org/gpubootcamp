# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.
GPUS_PER_NODE=8 # <--- remember to change the number of GPUs you actually have in your system
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1 #<-- currently we are using 1 node multigpus
NODE_RANK=0
WORLD_SIZE=8 # <--- remember to change the number of GPUs you actually have in your system

CHECKPOINT_PATH='./Megatron-LM/sv_ckpt/'
DATA_PATH='../dataset/EN/NVblogs_text_document'
VOCAB_FILE='../dataset/EN/50k/gpt2-vocab.json'
MERGE_FILE='../dataset/EN/50k/gpt2-merges.txt'
PROFILE_OUTPUT_PATH='/home/zcharpy/profiles/DLprof/naive/' # modify this to your own profile path



DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

nsys profile --stats=true --force-overwrite=true --duration=300 --trace=cudnn,cuda,osrt -o /home/zcharpy/profiles/GPT360M_naive \
    python -m torch.distributed.launch $DISTRIBUTED_ARGS \
    ./Megatron-LM/pretrain_gpt.py \
       --num-layers 16 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 1 \
       --global-batch-size 8 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-samples 100 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 10 \
       --save-interval 100 \
       --eval-interval 100 \
       --eval-iters 10 