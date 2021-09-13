# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.
NLAYERS=32
NHIDDEN=2048
NHEADS=32
SEQ_LEN=64
VOCAB_SIZE=32000

MODEL_SIZE=$((($NLAYERS * (12*$NHIDDEN**2 + 13*$NHIDDEN) + ($VOCAB_SIZE * $NHIDDEN) + ($SEQ_LEN * $NHIDDEN) ) / 10**9))
EXACT_MODEL_SIZE=$(($NLAYERS * (12*$NHIDDEN**2 + 13*$NHIDDEN) + ($VOCAB_SIZE * $NHIDDEN) + ($SEQ_LEN * $NHIDDEN) ))

echo $MODEL_SIZE
echo $EXACT_MODEL_SIZE