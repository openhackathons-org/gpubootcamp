# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved. 

FC := nvfortran
FLAGS := -O3 -w 
ACCFLAGS := -Minfo=accel
NVTXLIB := -lnvhpcwrapnvtx

rdf: rdf.f90
	${FC} ${FLAGS} ${ACCFLAGS} rdf.f90 -o rdf ${NVTXLIB} 
clean:
	rm -f *.o rdf
