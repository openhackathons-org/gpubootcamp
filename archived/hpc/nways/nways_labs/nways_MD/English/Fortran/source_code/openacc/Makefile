# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved. 

FC := nvfortran
FLAGS := -O3 -w 
ACCFLAGS := -Minfo=accel
NVTXLIB := -lnvhpcwrapnvtx

rdf: rdf.f90
	${FC} ${FLAGS} ${ACCFLAGS} -o rdf rdf.f90 ${NVTXLIB} 

clean:
	rm -f *.o rdf
