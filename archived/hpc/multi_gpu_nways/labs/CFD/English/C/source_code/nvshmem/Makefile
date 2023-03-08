# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
NP ?= 1
NVCC=nvcc
MPIRUN ?= mpirun
CUDA_HOME ?= /usr/local/cuda
ifndef NVSHMEM_HOME
$(error NVSHMEM_HOME is not set)
endif
ifndef MPI_HOME
$(error MPI_HOME is not set)
endif
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM70) $(GENCODE_SM80)

NVCC_FLAGS += -dc -Xcompiler -fopenmp -lineinfo -lnvToolsExt $(GENCODE_FLAGS) -std=c++14 -I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include
NVCC_LDFLAGS = -ccbin=mpic++ -L$(NVSHMEM_HOME)/lib -lnvshmem -L$(MPI_HOME)/lib -lmpi -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lnvToolsExt

left_shift: Makefile left_shift.cu
	$(NVCC) $(NVCC_FLAGS) left_shift.cu -c -o left_shift.o
	$(NVCC) $(GENCODE_FLAGS) left_shift.o -o left_shift $(NVCC_LDFLAGS)

jacobi_nvshmem: Makefile jacobi_nvshmem.cu
	$(NVCC) $(NVCC_FLAGS) jacobi_nvshmem.cu -c -o jacobi_nvshmem.o
	$(NVCC) $(GENCODE_FLAGS) jacobi_nvshmem.o -o jacobi_nvshmem $(NVCC_LDFLAGS)

.PHONY.: clean
clean:
	rm -rf jacobi_nvshmem left_shift *.o *.qdrep *.sqlite
