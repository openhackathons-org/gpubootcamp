# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved. 

CC := nvc++
CFLAGS := -O3 -w -ldl
ACCFLAGS := -Minfo=accel
NVTXLIB := -I/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/cuda/11.6/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/cuda/11.6/lib64 -lnvToolsExt

rdf: rdf.cpp
	${CC} ${CFLAGS} ${ACCFLAGS} -o rdf rdf.cpp ${NVTXLIB} 

clean:
	rm -f *.o rdf