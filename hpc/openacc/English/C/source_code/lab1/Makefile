
CC := pgcc
ACCFLAGS_1 := -fast
ACCFLAGS_2 := -fast -ta=multicore -Minfo=accel
ACCFLAGS_3 := -fast -ta=tesla:managed -Minfo=accel

laplace_serial: jacobi.c laplace2d.c
	${CC} ${ACCFLAGS_1} -o laplace jacobi.c laplace2d.c

laplace_multicore: jacobi.c laplace2d.c
	${CC} ${ACCFLAGS_2} -o laplace jacobi.c laplace2d.c

laplace_gpu: jacobi.c laplace2d.c
	${CC} ${ACCFLAGS_3} -o laplace jacobi.c laplace2d.c

clean:
	rm -f *.o laplace