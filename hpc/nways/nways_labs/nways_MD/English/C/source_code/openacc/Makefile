# Copyright (c) 2021 NVIDIA Corporation.  All rights reserved. 

CC := nvc++
CFLAGS := -O3 -w 
ACCFLAGS := -ta=tesla:managed,lineinfo -Minfo=accel

rdf: rdf.cpp
	${CC} ${CFLAGS} ${ACCFLAGS} -o rdf rdf.cpp 

clean:
	rm -f *.o rdf
