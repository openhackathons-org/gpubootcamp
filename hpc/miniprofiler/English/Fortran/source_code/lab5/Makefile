FC := pgf90
FFLAGS := -fast
LDFLAGS := -lnvToolsExt -lnetcdff
ACCFLAGS := -ta=tesla:managed -Minfo=accel
NVTXINC := -I/opt/pgi/linux86-64-llvm/2019/cuda/10.1/include/
NVTXLIB := -L/opt/pgi/linux86-64-llvm/2019/cuda/10.0/lib64/

miniWeather: nvtx.f90 miniWeather_openacc.f90
	$(FC) $(FFLAGS) $(ACCFLAGS) $(NVTXINC) nvtx.f90 miniWeather_openacc.f90 -o miniWeather $(NVTXLIB) $(LDFLAGS)

clean:
	rm -f *.o miniWeather
