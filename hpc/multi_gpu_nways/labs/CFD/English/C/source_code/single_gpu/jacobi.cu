/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <sstream>

#include <omp.h>
#include <nvToolsExt.h>

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 32

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus)                                                      \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
    }

constexpr float tol = 1.0e-8;

const float PI = 2.0 * std::asin(1.0);

__global__ void initialize_boundaries(float*  a_new, float*  a, const float pi, const int offset, 
					const int nx, const int my_ny, const int ny) {
    for (int iy = blockIdx.x * blockDim.x + threadIdx.x; iy < my_ny; iy += blockDim.x * gridDim.x) {
        const float y0 = sin(2.0 * pi * (offset + iy) / (ny - 1));
        a[iy * nx + 0] = y0;
        a[iy * nx + (nx - 1)] = y0;
        a_new[iy * nx + 0] = y0;
        a_new[iy * nx + (nx - 1)] = y0;
    }
}

__global__ void jacobi_kernel(float*  a_new, const float*  a, float*  l2_norm, const int iy_start,
                              const int iy_end, const int nx) {
    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    __shared__ float block_l2_sum[BLOCK_DIM_X*BLOCK_DIM_Y];
    unsigned thread_index = threadIdx.y*BLOCK_DIM_X + threadIdx.x;

    if (iy < iy_end && ix < (nx - 1)) {
	// Update grid point
        const float new_val = 0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                     a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
        a_new[iy * nx + ix] = new_val;
	float residue = new_val - a[iy * nx + ix];
	// Set block-level L2 norm value for this grid point
	block_l2_sum[thread_index] = residue * residue;
    }
    else {
	block_l2_sum[thread_index] = 0;
    }
    // Reduce L2 norm for the block in parallel
    for (unsigned stride = 1; stride < BLOCK_DIM_X*BLOCK_DIM_Y; stride *= 2) {
	__syncthreads();
	if ((thread_index) % (2*stride) == 0) {
    	    block_l2_sum[thread_index] += block_l2_sum[thread_index + stride];
	}
    }
    // Atomically update global L2 norm with block-reduced L2 norm
    if (thread_index == 0) {
	atomicAdd(l2_norm, block_l2_sum[0]);
    }
}

int get_argval(char** begin, char** end, const std::string& arg, const int default_val) {
    int argval = default_val;
    char** itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        std::istringstream inbuf(*itr);
        inbuf >> argval;
    }
    return argval;
}

double single_gpu(const int nx, const int ny, const int iter_max, float* const a_ref_h);

int main(int argc, char* argv[]) {
    const int iter_max = get_argval(argv, argv + argc, "-niter", 1000);
    const int nx = get_argval(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval(argv, argv + argc, "-ny", 16384);

    CUDA_RT_CALL(cudaSetDevice(0));
    CUDA_RT_CALL(cudaFree(0));

    float* a_ref_h;
    CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * sizeof(float)));
    
    double runtime_serial = single_gpu(nx, ny, iter_max, a_ref_h);

    printf("%dx%d: 1 GPU: %8.4f s\n", ny, nx, runtime_serial);

    return 0;
}

double single_gpu(const int nx, const int ny, const int iter_max, float* const a_ref_h) {
    float* a;
    float* a_new;

    float* l2_norm_d;
    float* l2_norm_h;

    int iy_start = 1;
    int iy_end = (ny - 1);

    CUDA_RT_CALL(cudaMalloc(&a, nx * ny * sizeof(float)));
    CUDA_RT_CALL(cudaMalloc(&a_new, nx * ny * sizeof(float)));

    CUDA_RT_CALL(cudaMemset(a, 0, nx * ny * sizeof(float)));
    CUDA_RT_CALL(cudaMemset(a_new, 0, nx * ny * sizeof(float)));

    // Set diriclet boundary conditions on left and right boarder
    nvtxRangePush("Init boundaries");
    initialize_boundaries<<<ny / 128 + 1, 128>>>(a, a_new, PI, 0, nx, ny, ny);
    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());
    nvtxRangePop();

    CUDA_RT_CALL(cudaMalloc(&l2_norm_d, sizeof(float)));
    CUDA_RT_CALL(cudaMallocHost(&l2_norm_h, sizeof(float)));

    CUDA_RT_CALL(cudaDeviceSynchronize());

    printf("Single GPU jacobi relaxation: %d iterations on %d x %d mesh\n", iter_max, ny, nx);

    dim3 dim_grid((nx + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (ny + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y, 1);
    dim3 dim_block(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

    int iter = 0;
    float l2_norm = 1.0;

    double start = omp_get_wtime();
    nvtxRangePush("Jacobi Solve");
    while (l2_norm > tol && iter < iter_max) {
        CUDA_RT_CALL(cudaMemset(l2_norm_d, 0, sizeof(float)));

	// Compute grid points for this iteration
        jacobi_kernel<<<dim_grid, dim_block>>>(a_new, a, l2_norm_d, iy_start, iy_end, nx);
       	CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaMemcpy(l2_norm_h, l2_norm_d, sizeof(float), cudaMemcpyDeviceToHost));

        // Apply periodic boundary conditions

        CUDA_RT_CALL(cudaMemcpy(a_new, a_new + (iy_end - 1) * nx, nx * sizeof(float),
                                     cudaMemcpyDeviceToDevice));
        CUDA_RT_CALL(cudaMemcpy(a_new + iy_end * nx, a_new + iy_start * nx, nx * sizeof(float),
                                     cudaMemcpyDeviceToDevice));

	CUDA_RT_CALL(cudaDeviceSynchronize());
	l2_norm = *l2_norm_h;
	l2_norm = std::sqrt(l2_norm);

        iter++;
	if ((iter % 100) == 0) printf("%5d, %0.6f\n", iter, l2_norm);

        std::swap(a_new, a);
    }
    nvtxRangePop();
    double stop = omp_get_wtime();

    CUDA_RT_CALL(cudaMemcpy(a_ref_h, a, nx * ny * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_RT_CALL(cudaFreeHost(l2_norm_h));
    CUDA_RT_CALL(cudaFree(l2_norm_d));

    CUDA_RT_CALL(cudaFree(a_new));
    CUDA_RT_CALL(cudaFree(a));
    return (stop - start);
}

