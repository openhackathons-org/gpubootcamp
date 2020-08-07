
FC := pgfortran
ACCFLAGS_1 := -fast -ta=tesla -Minfo=accel
ACCFLAGS_2 := -fast -ta=tesla:managed -Minfo=accel

laplace_update: laplace2d.f90 jacobi.f90
	${FC} ${ACCFLAGS_1} -o laplace_update laplace2d.f90 jacobi.f90

laplace_no_update: laplace2d.f90 jacobi.f90
	${FC} ${ACCFLAGS_1} -o laplace_no_update laplace2d.f90 jacobi.f90

laplace_no_managed: laplace2d.f90 jacobi.f90
	${FC} ${ACCFLAGS_1} -o laplace laplace2d.f90 jacobi.f90

laplace_managed: laplace2d.f90 jacobi.f90
	${FC} ${ACCFLAGS_2} -o laplace_managed laplace2d.f90 jacobi.f90

clean:
	rm -f *.o laplace laplace_*