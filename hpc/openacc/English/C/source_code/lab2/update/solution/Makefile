
CC := pgcc
ACCFLAGS_1 := -fast -ta=tesla -Minfo=accel
ACCFLAGS_2 := -fast -ta=tesla:managed -Minfo=accel

laplace_update: jacobi.c laplace2d.c
	${CC} ${ACCFLAGS_1} -o laplace_update jacobi.c laplace2d.c

laplace_no_update: jacobi.c laplace2d.c
	${CC} ${ACCFLAGS_1} -o laplace_no_update jacobi.c laplace2d.c

laplace_no_managed: jacobi.c laplace2d.c
	${CC} ${ACCFLAGS_1} -o laplace jacobi.c laplace2d.c

laplace_managed: jacobi.c laplace2d.c
	${CC} ${ACCFLAGS_2} -o laplace_managed jacobi.c laplace2d.c

clean:
	rm -f *.o laplace laplace_*