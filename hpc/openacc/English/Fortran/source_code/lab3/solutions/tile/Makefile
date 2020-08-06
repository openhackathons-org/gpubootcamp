
FC := pgfortran
ACCFLAGS:= -fast -ta=tesla:cc70 -Minfo=accel

laplace: laplace2d.f90 jacobi.f90
	${FC} ${ACCFLAGS} -o laplace laplace2d.f90 jacobi.f90

clean:
	rm -f *.o laplace 