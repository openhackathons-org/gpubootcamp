# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved. 

FC := nvfortran
FLAGS := -O3 -w -ldl
ACCFLAGS := -Minfo=accel
NVTXLIB := -I/opt/nvidia/hpc_sdk/Linux_x86_64/20.11/cuda/11.0/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/20.11/cuda/11.0/lib64 -lnvToolsExt

rdf:nvtx.f90 rdf.f90
	${FC} ${FLAGS} ${ACCFLAGS} nvtx.f90 rdf.f90 -o rdf ${NVTXLIB} 
clean:
	rm -f *.o rdf
