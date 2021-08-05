/* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#include <cstdio>

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

void launch_initialize_boundaries(float*  a_new, float*  a, const float pi, const int offset, 
                                    const int nx, const int my_ny, const int ny) {
    initialize_boundaries<<<my_ny / 128 + 1, 128>>>(a_new, a, pi, offset, nx, my_ny, ny);
}

void launch_jacobi_kernel(float*  a_new, const float*  a, float*  l2_norm, const int iy_start,
                              const int iy_end, const int nx, cudaStream_t stream) {
    dim3 dim_block(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    dim3 dim_grid((nx + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
                  ((iy_end - iy_start) + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y, 1);
    jacobi_kernel<<<dim_grid, dim_block, 0, stream>>>(a_new, a, l2_norm, iy_start, iy_end, nx);
}

