# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved. 

CC := nvc++
CFLAGS := -O3 -w
ACCFLAGS := -ta=tesla:managed -Minfo=accel

miniWeather: miniWeather_openacc.cpp
	${CC} ${CFLAGS} ${ACCFLAGS} -o miniWeather miniWeather_openacc.cpp 

clean:
	rm -f *.o miniWeather
