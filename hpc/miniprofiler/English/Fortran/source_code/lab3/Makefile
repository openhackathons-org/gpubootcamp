# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved. 
FC := pgf90
FCC := pgfortran
FFLAGS := -fast
LDFLAGS := -lnvToolsExt 
ACCFLAGS := -ta=tesla:managed -Minfo=accel
NVTXINC := -I/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/cuda/11.0/include
NVTXLIB := -L/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/cuda/11.0/lib64/

miniWeather: nvtx.f90 miniWeather_openacc.f90
	$(FC) $(FFLAGS) $(ACCFLAGS) $(NVTXINC) nvtx.f90 miniWeather_openacc.f90 -o miniWeather $(NVTXLIB) $(LDFLAGS)

clean:
	rm -f *.o miniWeather
