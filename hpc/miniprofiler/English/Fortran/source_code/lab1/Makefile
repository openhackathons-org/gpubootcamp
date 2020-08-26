# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved. 
FC := pgf90
FCC := pgfortran
FFLAGS := -fast
LDFLAGS := -lnvToolsExt 
ACCFLAGS := -Minfo=accel
NVTXINC := -I/opt/pgi/linux86-64-llvm/2019/cuda/10.1/include/
NVTXLIB := -L/opt/pgi/linux86-64-llvm/2019/cuda/10.0/lib64/

miniWeather: nvtx.f90 miniWeather_serial.f90
	$(FC) $(FFLAGS) $(ACCFLAGS) $(NVTXINC) nvtx.f90 miniWeather_serial.f90 -o miniWeather $(NVTXLIB) $(LDFLAGS)

clean:
	rm -f *.o miniWeather
