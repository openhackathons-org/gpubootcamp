CC := pgc++
CFLAGS := -O3 -w
ACCFLAGS := -ta=tesla:managed -Minfo=accel
LDFLAGS := -lnetcdf

miniWeather: miniWeather_openacc.cpp
	${CC} ${CFLAGS} ${ACCFLAGS} -o miniWeather miniWeather_openacc.cpp ${LDFLAGS}

clean:
	rm -f *.o miniWeather
