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
#include <cmath>
#include <cstdio>
#include <iostream>
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

constexpr int MAX_NUM_DEVICES = 32;

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

int get_parsed_vals(char** begin, char **end, int* devices,
		const std::string& arg, const int default_val) {
    int numGPUs = default_val;
    char** itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        numGPUs = 0;
        std::string dev_ids(*itr);
	int currpos = 0, nextpos = 0;
	do {
	    nextpos = dev_ids.find_first_of(",", currpos);
            devices[numGPUs] = stoi(dev_ids.substr(currpos, nextpos));
	    numGPUs++;
	    currpos = nextpos + 1;
        } while (nextpos != std::string::npos);
    }
    else {
        for (int i = 0; i < numGPUs; i++) {
            devices[i] = i;
	}
    }
    return numGPUs;
}

bool get_arg(char** begin, char** end, const std::string& arg) {
    char** itr = std::find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}

double single_gpu(const int nx, const int ny, const int iter_max, float* const a_ref_h);

int main(int argc, char* argv[]) {
    const int iter_max = get_argval(argv, argv + argc, "-niter", 1000);
    const int nx = get_argval(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval(argv, argv + argc, "-ny", 16384);
    const bool p2p = get_arg(argv, argv + argc, "-p2p");
    
    // Get GPU mapping from runtime arguments
    int available_devices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&available_devices));
    int devices[MAX_NUM_DEVICES];
    int num_devices = get_parsed_vals(argv, argv + argc, devices, "-gpus", available_devices);

    float* a[MAX_NUM_DEVICES];
    float* a_new[MAX_NUM_DEVICES];
    float* a_ref_h;
    float* a_h;
    double runtime_serial = 0.0;

    float* l2_norm_d[MAX_NUM_DEVICES];
    float* l2_norm_h[MAX_NUM_DEVICES];

    int iy_start[MAX_NUM_DEVICES];
    int iy_end[MAX_NUM_DEVICES];

    int chunk_size[MAX_NUM_DEVICES];

    // Compute chunk size and allocate memory on GPUs
    for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
        CUDA_RT_CALL(cudaSetDevice(devices[dev_id]));
        CUDA_RT_CALL(cudaFree(0));

        if (0 == dev_id) {
	        // Allocate memory on host and record single-GPU timings
            CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * sizeof(float)));
            CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * sizeof(float)));
            runtime_serial = single_gpu(nx, ny, iter_max, a_ref_h);
        }

        // ny - 2 rows are distributed amongst `size` ranks in such a way
        // that each rank gets either (ny - 2) / size or (ny - 2) / size + 1 rows.
        // This optimizes load balancing when (ny - 2) % size != 0
        int chunk_size_low = (ny - 2) / num_devices;
        int chunk_size_high = chunk_size_low + 1;

        // To calculate the number of ranks that need to compute an extra row,
        // the following formula is derived from this equation:
        // num_ranks_low * chunk_size_low + (size - num_ranks_low) * (chunk_size_low + 1) = (ny - 2)
        int num_ranks_low = num_devices * chunk_size_low + num_devices - (ny - 2);  

        if (dev_id < num_ranks_low)
            chunk_size[dev_id] = chunk_size_low;
        else
            chunk_size[dev_id] = chunk_size_high;

	    // Allocate memory on GPU
        CUDA_RT_CALL(cudaMalloc(a + dev_id, nx * (chunk_size[dev_id] + 2) * sizeof(float)));
        CUDA_RT_CALL(cudaMalloc(a_new + dev_id, nx * (chunk_size[dev_id] + 2) * sizeof(float)));

        CUDA_RT_CALL(cudaMemset(a[dev_id], 0, nx * (chunk_size[dev_id] + 2) * sizeof(float)));
        CUDA_RT_CALL(cudaMemset(a_new[dev_id], 0, nx * (chunk_size[dev_id] + 2) * sizeof(float)));

        // Calculate local domain boundaries
        int iy_start_global;  // My start index in the global array
        if (dev_id < num_ranks_low) {
            iy_start_global = dev_id * chunk_size_low + 1;
        } else {
            iy_start_global =
                num_ranks_low * chunk_size_low + (dev_id - num_ranks_low) * chunk_size_high + 1;
        }

        iy_start[dev_id] = 1;
        iy_end[dev_id] = iy_start[dev_id] + chunk_size[dev_id];

        // Set dirichlet boundary conditions on left and right boarder
        initialize_boundaries<<<(ny / num_devices) / 128 + 1, 128>>>(
            a[dev_id], a_new[dev_id], PI, iy_start_global - 1, nx, (chunk_size[dev_id] + 2), ny);
        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize());

        CUDA_RT_CALL(cudaMalloc(l2_norm_d + dev_id, sizeof(float)));
        CUDA_RT_CALL(cudaMallocHost(l2_norm_h + dev_id, sizeof(float)));

        if (p2p == true) {
            const int top = dev_id > 0 ? dev_id - 1 : (num_devices - 1);
            int canAccessPeer = 0;
            // TODO: Part 2- Check whether GPU "devices[dev_id]" can access peer "devices[top]"
            CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, /*Fill me*/, /*Fill me*/));
            if (canAccessPeer) {
            // TODO: Part 2- Enable peer access from GPU "devices[dev_id]" to "devices[top]"
                CUDA_RT_CALL(cudaDeviceEnablePeerAccess(/*Fill me*/, 0));
            }
            const int bottom = (dev_id + 1) % num_devices;
            if (top != bottom) {
                canAccessPeer = 0;
                // TODO: Part 2- Check and enable peer access from GPU "devices[dev_id]" to
                // "devices[bottom]", whenever possible
                CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, /*Fill me*/, /*Fill me*/));
                if (canAccessPeer) {
                    CUDA_RT_CALL(cudaDeviceEnablePeerAccess(/*Fill me*/, 0));
                }
            }
        }
        CUDA_RT_CALL(cudaDeviceSynchronize());
    }

    // Share initial top and bottom local grid-point values between neighbours
    for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
        CUDA_RT_CALL(cudaSetDevice(devices[dev_id]));
        const int top = dev_id > 0 ? dev_id - 1 : (num_devices - 1);
        const int bottom = (dev_id + 1) % num_devices;
        CUDA_RT_CALL(cudaMemcpy(a_new[top] + (iy_end[top] * nx),
                     a_new[dev_id] + iy_start[dev_id] * nx, nx * sizeof(float),
                     cudaMemcpyDeviceToDevice));
        CUDA_RT_CALL(cudaMemcpy(a_new[bottom], a_new[dev_id] + (iy_end[dev_id] - 1) * nx,
                     nx * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
        CUDA_RT_CALL(cudaSetDevice(devices[dev_id]));
        CUDA_RT_CALL(cudaDeviceSynchronize());
    }

    printf("Jacobi relaxation: %d iterations on %d x %d mesh\n", iter_max, nx, ny);

    dim3 dim_block(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    int iter = 0;
    float l2_norm = 1.0;

    double start = omp_get_wtime();
    nvtxRangePush("Jacobi solve");
    while (l2_norm > tol && iter < iter_max) {
	    // Launch device kernel on each GPU
        for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
            // TODO: Part 1- Set current GPU to be "devices[dev_id]"
            CUDA_RT_CALL(cudaSetDevice(/*Fill me*/));

            CUDA_RT_CALL(cudaMemsetAsync(l2_norm_d[dev_id], 0, sizeof(float)));
            dim3 dim_grid((nx + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
                          (chunk_size[dev_id] + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y, 1);
	    
            // TODO: Part 1- Call Jacobi kernel with "dim_grid" blocks in grid and "dim_block"
            // blocks per thread. "dev_id" variable points to corresponding memory allocated 
            // for the current GPU.
            jacobi_kernel<<</*Fill me*/, /*Fill me*/>>>(/*Fill me*/);

            // TODO: Part 1- Copy GPU-local L2 norm "l2_norm_d" back to CPU "l2_norm_h"
            CUDA_RT_CALL(cudaMemcpyAsync(/*Fill me*/, /*Fill me*/, sizeof(float), /*Fill me*/));
	}
    // Launch async memory copy operations for halo exchange and 
	// for copying local-grid L2 norm from each GPU to host
	for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
            const int top = dev_id > 0 ? dev_id - 1 : (num_devices - 1);
            const int bottom = (dev_id + 1) % num_devices;
            
            // TODO: Part 1- Set current GPU
            CUDA_RT_CALL(cudaSetDevice(/*Fill me*/));

            // TODO: Part 1- Implement halo exchange with top neighbour "top"
            CUDA_RT_CALL(cudaMemcpyAsync(/*Fill me*/, /*Fill me*/, nx * sizeof(float), /*Fill me*/));
	    
            // TODO: Part 1- Implement halo exchange with bottom neighbour "bottom"
            CUDA_RT_CALL(cudaMemcpyAsync(/*Fill me*/, /*Fill me*/, nx * sizeof(float), /*Fill me*/));
        }
        l2_norm = 0.0;
        // Synchronize devices and compute global L2 norm
        for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
            // TODO: part 1- Set current GPU and call cudaDeviceSynchronize()
	        CUDA_RT_CALL(cudaSetDevice(/*Fill me*/));
            CUDA_RT_CALL(/*Fill me*/);

            l2_norm += *(l2_norm_h[dev_id]);
        }

        l2_norm = std::sqrt(l2_norm);
        
	iter++;
        if ((iter % 100) == 0) printf("%5d, %0.6f\n", iter, l2_norm);

        for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
            std::swap(a_new[dev_id], a[dev_id]);
        }
    }

    nvtxRangePop();
    double stop = omp_get_wtime();

    int offset = nx;
    // Copy computed grid back to host from each GPU
    for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
        CUDA_RT_CALL(
            cudaMemcpy(a_h + offset, a[dev_id] + nx,
                       std::min((nx * ny) - offset, nx * chunk_size[dev_id]) * sizeof(float),
                       cudaMemcpyDeviceToHost));
        offset += std::min(chunk_size[dev_id] * nx, (nx * ny) - offset);
    }

    // Compare against single GPU execution for correctness
    bool result_correct = true;
    for (int iy = 1; result_correct && (iy < (ny - 1)); ++iy) {
        for (int ix = 1; result_correct && (ix < (nx - 1)); ++ix) {
            if (std::fabs(a_ref_h[iy * nx + ix] - a_h[iy * nx + ix]) > tol) {
                fprintf(stderr,
                        "ERROR: a[%d * %d + %d] = %f does not match %f "
                        "(reference)\n",
                        iy, nx, ix, a_h[iy * nx + ix], a_ref_h[iy * nx + ix]);
                result_correct = false;
            }
        }
    }

    if (result_correct) {
        printf("Num GPUs: %d. Using GPU ID: ", num_devices);
	for (int i = 0; i < num_devices; i++) {
            printf("%d, ", devices[i]);
	}
        printf(
	    "\n%dx%d: 1 GPU: %8.4f s, %d GPUs: %8.4f s, speedup: %8.2f, "
            "efficiency: %8.2f \n",
            ny, nx, runtime_serial, num_devices, (stop - start),
            runtime_serial / (stop - start),
            runtime_serial / (num_devices * (stop - start)) * 100);
    }

    for (int dev_id = (num_devices - 1); dev_id >= 0; --dev_id) {
        CUDA_RT_CALL(cudaSetDevice(dev_id));

        CUDA_RT_CALL(cudaFreeHost(l2_norm_h[dev_id]));
        CUDA_RT_CALL(cudaFree(l2_norm_d[dev_id]));

        CUDA_RT_CALL(cudaFree(a_new[dev_id]));
        CUDA_RT_CALL(cudaFree(a[dev_id]));
        if (0 == dev_id) {
            CUDA_RT_CALL(cudaFreeHost(a_h));
            CUDA_RT_CALL(cudaFreeHost(a_ref_h));
        }
    }

    return result_correct ? 0 : 1;
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

    printf("Single GPU jacobi relaxation: %d iterations on %d x %d mesh\n", iter_max, nx, ny);

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

