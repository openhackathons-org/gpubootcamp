# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
NVCC=nvcc
# CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/21.5/cuda/
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM70) $(GENCODE_SM80)
NVCC_FLAGS += -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -lnvToolsExt $(GENCODE_FLAGS) -std=c++14

jacobi_memcpy: jacobi_memcpy.cu
	$(NVCC) $(NVCC_FLAGS) jacobi_memcpy.cu -o jacobi_memcpy

jacobi_streams: jacobi_streams.cu
	$(NVCC) $(NVCC_FLAGS) jacobi_streams.cu -o jacobi_streams

jacobi_streams_events: jacobi_streams_events.cu
	$(NVCC) $(NVCC_FLAGS) jacobi_streams_events.cu -o jacobi_streams_events

all: jacobi_memcpy jacobi_streams jacobi_streams_events

.PHONY: clean
clean:
	rm -f jacobi_memcpy jacobi_streams jacobi_streams_events *.qdrep *.sqlite *.nsys-rep

