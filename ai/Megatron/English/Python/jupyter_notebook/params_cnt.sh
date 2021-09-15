# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.
NLAYERS=32 ## modify this param
NHIDDEN=4096 ## modify this param
NHEADS=32 ## modify this param
SEQ_LEN=512 ## modify this param
VOCAB_SIZE=56000 ## modify this param

MODEL_SIZE=$((($NLAYERS * (12*$NHIDDEN**2 + 13*$NHIDDEN) + ($VOCAB_SIZE * $NHIDDEN) + ($SEQ_LEN * $NHIDDEN) ) / 10**9))
EXACT_MODEL_SIZE=$(($NLAYERS * (12*$NHIDDEN**2 + 13*$NHIDDEN) + ($VOCAB_SIZE * $NHIDDEN) + ($SEQ_LEN * $NHIDDEN) ))

echo $MODEL_SIZE
echo $EXACT_MODEL_SIZE