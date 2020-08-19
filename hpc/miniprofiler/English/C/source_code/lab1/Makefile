CC := pgc++
CFLAGS := -O3 -w -ldl
ACCFLAGS := -Minfo=accel
NVTXLIB := -I/opt/pgi/linux86-64-llvm/2019/cuda/10.1/include/

miniWeather: miniWeather_serial.cpp
	${CC} ${CFLAGS} ${ACCFLAGS} -o miniWeather miniWeather_serial.cpp ${NVTXLIB} 

clean:
	rm -f *.o miniWeather
