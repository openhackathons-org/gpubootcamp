# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved. 

CC := nvc++
CFLAGS := -O3 -w -ldl
ACCFLAGS := -Minfo=accel
NVTXLIB := -I/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/cuda/11.6/include

miniWeather: miniWeather_serial.cpp
	${CC} ${CFLAGS} ${ACCFLAGS} -o miniWeather miniWeather_serial.cpp ${NVTXLIB} 

clean:
	rm -f *.o miniWeather
