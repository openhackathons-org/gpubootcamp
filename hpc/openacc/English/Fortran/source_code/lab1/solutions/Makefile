# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved. 
FC := pgfortran
ACCFLAGS_1 := -fast
ACCFLAGS_2 := -fast -ta=multicore -Minfo=accel
ACCFLAGS_3 := -fast -ta=tesla:managed -Minfo=accel

laplace_serial: laplace2d.f90 jacobi.f90 
	${FC} ${ACCFLAGS_1} -o laplace laplace2d.f90 jacobi.f90

laplace_multicore: laplace2d.f90 jacobi.f90 
	${FC} ${ACCFLAGS_2} -o laplace laplace2d.f90 jacobi.f90 

laplace_gpu: laplace2d.f90 jacobi.f90 
	${FC} ${ACCFLAGS_3} -o laplace laplace2d.f90 jacobi.f90 

clean:
	rm -f *.o laplace