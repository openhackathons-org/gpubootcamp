# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.
NLAYERS=  ## modify this param
NHIDDEN=  ## modify this param
NHEADS=  ## modify this param
SEQ_LEN= ## modify this param
VOCAB_SIZE=56000 ## modify this param

MODEL_SIZE=$((($NLAYERS * (12*$NHIDDEN**2 + 13*$NHIDDEN) + ($VOCAB_SIZE * $NHIDDEN) + ($SEQ_LEN * $NHIDDEN) ) / 10**9))
EXACT_MODEL_SIZE=$(($NLAYERS * (12*$NHIDDEN**2 + 13*$NHIDDEN) + ($VOCAB_SIZE * $NHIDDEN) + ($SEQ_LEN * $NHIDDEN) ))

echo $MODEL_SIZE
echo $EXACT_MODEL_SIZE