# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
NVCC=nvcc
MPICXX=mpicxx
#CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/21.5/cuda/
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 
GENCODE_FLAGS	:= $(GENCODE_SM70) $(GENCODE_SM80)

NVCC_FLAGS += -lineinfo $(GENCODE_FLAGS) -std=c++14
MPICXX_FLAGS = -g -I$(CUDA_HOME)/include  -fopenmp -std=c++14
LD_FLAGS = -L$(CUDA_HOME)/lib64 -lcudart -lnvToolsExt

hello_world: Makefile hello_world.c
	$(MPICXX) $(MPICXX_FLAGS) hello_world.c $(LD_FLAGS) -o hello_world

jacobi_memcpy_mpi: Makefile jacobi_memcpy_mpi.cpp jacobi_kernels.o
	$(MPICXX) $(MPICXX_FLAGS) jacobi_memcpy_mpi.cpp jacobi_kernels.o $(LD_FLAGS) -o jacobi_memcpy_mpi

jacobi_cuda_aware_mpi: Makefile jacobi_cuda_aware_mpi.cpp jacobi_kernels.o
	$(MPICXX) $(MPICXX_FLAGS) jacobi_cuda_aware_mpi.cpp jacobi_kernels.o $(LD_FLAGS) -o jacobi_cuda_aware_mpi

jacobi_kernels.o: Makefile jacobi_kernels.cu
	$(NVCC) $(NVCC_FLAGS) jacobi_kernels.cu -c

all: hello_world jacobi_memcpy_mpi jacobi_cuda_aware_mpi

.PHONY.: clean
clean:
	rm -rf hello_world jacobi_memcpy_mpi jacobi_cuda_aware_mpi *.o *.qdrep *.sqlite

