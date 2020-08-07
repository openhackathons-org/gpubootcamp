
CC := pgcc
ACCFLAGS:= -fast -ta=tesla -Minfo=accel

laplace: jacobi.c laplace2d.c
	${CC} ${ACCFLAGS} -o laplace jacobi.c laplace2d.c

clean:
	rm -f *.o laplace 