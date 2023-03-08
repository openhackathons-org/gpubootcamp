# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
NVCC=nvcc
MPICXX=mpicxx
MPIRUN ?= mpirun
#CUDA_HOME ?= /usr/local/cuda
#NCCL_HOME ?= /usr/nccl/
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80
GENCODE_FLAGS	:= $(GENCODE_SM70) $(GENCODE_SM80)

NVCC_FLAGS += -lineinfo $(GENCODE_FLAGS) -std=c++14
MPICXX_FLAGS = -DUSE_NVTX -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include -fopenmp -std=c++14
LD_FLAGS = -L$(CUDA_HOME)/lib64 -lcudart -lnvToolsExt -lnccl

jacobi_nccl: Makefile jacobi_nccl.cpp jacobi_kernels.o
	$(MPICXX) $(MPICXX_FLAGS) jacobi_nccl.cpp jacobi_kernels.o $(LD_FLAGS) -o jacobi_nccl

jacobi_kernels.o: Makefile jacobi_kernels.cu
	$(NVCC) $(NVCC_FLAGS) jacobi_kernels.cu -c

.PHONY.: clean
clean:
	rm -rf jacobi_nccl jacobi_kernels.o *.qdrep *.sqlite

