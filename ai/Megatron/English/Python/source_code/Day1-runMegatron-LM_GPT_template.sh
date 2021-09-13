#!/bin/bash
# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.

#SBATCH -t 12:00:00
#SBATCH -A berzelius-2021-43
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH -N 2
#SBATCH --gres=gpu:8
###  -----------------  modify <UserName> and <FILL_IN> in the section below -----------------
#SBATCH --output=//proj/guest_at_nsc/users/<UserName>/output/multinode_template_%x_%j_$DATETIME.log 

DIR='/proj/guest_at_nsc/users/<UserName>/'
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
CHECKPOINT_PATH=$DIR/output/sv_gpt3_ckpt/
VOCAB_FILE=$DIR/dataset/vocab.json
MERGE_FILE=$DIR/dataset/merges.txt
DATA_PATH=$DIR/dataset/SVGPT_32k_text_document

NHIDDEN=<FILL_IN>
NLAYERS=<FILL_IN>
NHEADS=<FILL_IN>
SEQ_LEN=<FILL_IN>
MAX_POS_EM=<FILL_IN>
### ----------------- end of section : do NOT modify anything else -----------------

VOCAB_SIZE=32000
MODEL_SIZE=$((($NLAYERS * (12*$NHIDDEN**2 + 13*$NHIDDEN) + ($VOCAB_SIZE * $NHIDDEN) + ($SEQ_LEN * $NHIDDEN) ) / 10**9))
EXACT_MODEL_SIZE=$(($NLAYERS * (12*$NHIDDEN**2 + 13*$NHIDDEN) + ($VOCAB_SIZE * $NHIDDEN) + ($SEQ_LEN * $NHIDDEN) ))

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
echo "LOCAL_PROCESS_RANK="$LOCAL_PROCESS_RANK
export MASTER_PORT=12340
export WORLD_SIZE=16
# The first hostname is the master address
#master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#master_addr=`perl -le '$_=$ENV{"SLURM_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
#master_addr=$(ip address show eth1 | grep -E '\<inet\>' | cut -d' ' -f6 | cut -c-10)
master_addr=MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
export MASTER_ADDR=$master_addr

echo "master_addr="$master_addr
echo "SLURM_NODEID="$SLURN_NODEID
echo "hostname="$(hostname)
echo "SLURM_PROCID="$SLURM_PROCID
echo "LOCAL_PROCESS_RANK="$LOCAL_PROCESS_RANK
echo "SLURM_LOCALID="$SLURM_LOCALID

# Execute my Singularity image binding in the current working directory
#export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export OMP_NUM_THREADS=1
#export NCCL_IB_HCA="^mlx5_4,mlx5_5,mlx5_10,mlx5_11"
export NCCL_NET=IB
export NCCL_IB_HCA=${UCX_NET_DEVICES} 
echo $NCCL_IB_HCA
export NODE_RANK=0
echo "NODE_RANK="$NODE_RANK

#DISTRIBUTED_ARGS="--nproc_per_node 8 --nnodes 2 --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
DISTRIBUTED_ARGS="--nproc_per_node 8"

options="--num-layers ${NLAYERS} \
			--hidden-size ${NHIDDEN} \
			--num-attention-heads ${NHEADS} \
			--seq-length ${SEQ_LEN} \
			--max-position-embeddings ${MAX_POS_EM} \
			--lr 0.00015 \
			--train-iters 100 \
			--min-lr 0.00001 \
			--lr-decay-iters 99 \
			--lr-warmup-fraction 0.01 \
			--override-lr-scheduler \
			--micro-batch-size 1 \
			--global-batch-size 16 \
			--vocab-file ${VOCAB_FILE} \
			--merge-file ${MERGE_FILE} \
			--split 949,50,1 \
			--distributed-backend nccl \
			--log-interval 10 \
			--save-interval 100 \
			--eval-interval 100 \
			--eval-iters 10 \
			--checkpoint-activations \
			--tensor-model-parallel-size 8 \
			--pipeline-model-parallel-size 2 \
			--save ${CHECKPOINT_PATH} \
			--load ${CHECKPOINT_PATH} \
			--data-path ${DATA_PATH} \
			--fp16 "
# Execute my Singularity image binding in the current working directory
cd /proj/guest_at_nsc/users/zcharpy/
export SINGULARITY_BINDPATH="/proj/guest_at_nsc/users/zcharpy/"
echo "SLURN_JOBID="$SLURM_JOBID 
export jobid=SLURM_JOBID 
# containing the Python script I want to execute

export SINGULARITY_BINDPATH="/proj/guest_at_nsc/users/zcharpy/"
# containing the Python script I want to execute
singularity exec --nv pytorch_21.03.sif python -m torch.distributed.launch ${DISTRIBUTED_ARGS} \
	${DIR}/Megatron-LM/pretrain_gpt.py \
	${options}
	 
echo $MODEL_SIZE
echo $EXACT_MODEL_SIZE
echo "if you see this line, this means that you have successfully ran Megatron-LM, congratulations !"