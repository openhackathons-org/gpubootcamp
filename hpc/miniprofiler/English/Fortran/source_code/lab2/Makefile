# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved. 
FC := nvfortran
FCC := nvfortran
FFLAGS := -fast
LDFLAGS := -lnvhpcwrapnvtx
ACCFLAGS := -ta=tesla:managed -Minfo=accel

miniWeather: miniWeather_openacc.f90
	$(FC) $(FFLAGS) $(ACCFLAGS) miniWeather_openacc.f90 -o miniWeather $(LDFLAGS)

clean:
	rm -f *.o miniWeather
