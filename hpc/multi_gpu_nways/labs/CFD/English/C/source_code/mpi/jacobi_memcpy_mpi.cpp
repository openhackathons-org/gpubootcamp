/* Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <mpi.h>
#include <omp.h>

#define MPI_CALL(call)                                                                \
    {                                                                                 \
        int mpi_status = call;                                                        \
        if (0 != mpi_status) {                                                        \
            char mpi_error_string[MPI_MAX_ERROR_STRING];                              \
            int mpi_error_string_length = 0;                                          \
            MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length); \
            if (NULL != mpi_error_string)                                             \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %s "                                                    \
                        "(%d).\n",                                                    \
                        #call, __LINE__, __FILE__, mpi_error_string, mpi_status);     \
            else                                                                      \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %d.\n",                                                 \
                        #call, __LINE__, __FILE__, mpi_status);                       \
        }                                                                             \
    }

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

void launch_initialize_boundaries(float* __restrict__ const a_new, float* __restrict__ const a,
                                  const float pi, const int offset, const int nx, const int my_ny,
                                  const int ny);

void launch_jacobi_kernel(float* __restrict__ const a_new, const float* __restrict__ const a,
                          float* __restrict__ const l2_norm, const int iy_start, const int iy_end,
                          const int nx);

double single_gpu(const int nx, const int ny, const int iter_max, 
                    float* const a_ref_h, bool print);

int get_argval(char** begin, char** end, const std::string& arg, const int default_val) {
    int argval = default_val;
    char** itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        std::istringstream inbuf(*itr);
        inbuf >> argval;
    }
    return argval;
}

bool get_arg(char** begin, char** end, const std::string& arg) {
    char** itr = std::find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}

int main(int argc, char* argv[]) {
    MPI_CALL(MPI_Init(&argc, &argv));
    int rank;
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    int size;
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    
    const int iter_max = get_argval(argv, argv + argc, "-niter", 1000);
    const int nx = get_argval(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval(argv, argv + argc, "-ny", 16384);
    const bool skip_single_gpu = get_arg(argv, argv + argc, "-skip_single_gpu");

    int local_rank = -1;
    // TODO: Part 1- Obtain the node-level local rank by splitting the global communicator
    // Free the local communicator after its use
    MPI_Comm local_comm;
    MPI_CALL(MPI_Comm_split_type(/*Fill me*/));

    MPI_CALL(MPI_Comm_rank(/*Fill me*/));

    MPI_CALL(MPI_Comm_free(/*Fill me*/));

    CUDA_RT_CALL(cudaSetDevice(local_rank % num_devices));
    CUDA_RT_CALL(cudaFree(0));

    float* a_ref_h;
    CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * sizeof(float)));
    float* a_h;
    CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * sizeof(float)));
    float* top_halo_buf;
    CUDA_RT_CALL(cudaMallocHost(&top_halo_buf, nx * sizeof(float)));
    float* bot_halo_buf;
    CUDA_RT_CALL(cudaMallocHost(&bot_halo_buf, nx * sizeof(float)));

    double runtime_serial = 1;
    if (!skip_single_gpu){
        runtime_serial = single_gpu(nx, ny, iter_max, a_ref_h, rank == 0);
    }

    // ny - 2 rows are distributed amongst `size` ranks in such a way
    // that each rank gets either (ny - 2) / size or (ny - 2) / size + 1 rows.
    // This optimizes load balancing when (ny - 2) % size != 0
    int chunk_size;
    int chunk_size_low = (ny - 2) / size;
    int chunk_size_high = chunk_size_low + 1;
    // To calculate the number of ranks that need to compute an extra row,
    // the following formula is derived from this equation:
    // num_ranks_low * chunk_size_low + (size - num_ranks_low) * (chunk_size_low + 1) = ny - 2
    int num_ranks_low = size * chunk_size_low + size -
                        (ny - 2);  // Number of ranks with chunk_size = chunk_size_low
    if (rank < num_ranks_low)
        chunk_size = chunk_size_low;
    else
        chunk_size = chunk_size_high;

    float* a;
    CUDA_RT_CALL(cudaMalloc(&a, nx * (chunk_size + 2) * sizeof(float)));
    float* a_new;
    CUDA_RT_CALL(cudaMalloc(&a_new, nx * (chunk_size + 2) * sizeof(float)));

    CUDA_RT_CALL(cudaMemset(a, 0, nx * (chunk_size + 2) * sizeof(float)));
    CUDA_RT_CALL(cudaMemset(a_new, 0, nx * (chunk_size + 2) * sizeof(float)));

    // Calculate local domain boundaries
    int iy_start_global;  // My start index in the global array
    if (rank < num_ranks_low) {
        iy_start_global = rank * chunk_size_low + 1;
    } else {
        iy_start_global =
            num_ranks_low * chunk_size_low + (rank - num_ranks_low) * chunk_size_high + 1;
    }
    int iy_end_global = iy_start_global + chunk_size - 1;  // My last index in the global array

    int iy_start = 1;
    int iy_end = iy_start + chunk_size;

    // Set diriclet boundary conditions on left and right boarder
    launch_initialize_boundaries(a, a_new, PI, iy_start_global - 1, nx, (chunk_size + 2), ny);
    CUDA_RT_CALL(cudaDeviceSynchronize());

    float* l2_norm_d;
    CUDA_RT_CALL(cudaMalloc(&l2_norm_d, sizeof(float)));
    float* l2_norm_h;
    CUDA_RT_CALL(cudaMallocHost(&l2_norm_h, sizeof(float)));

    CUDA_RT_CALL(cudaDeviceSynchronize());

    if (0 == rank) {
        printf("Jacobi relaxation: %d iterations on %d x %d mesh\n", iter_max, nx, ny);
    }

    int iter = 0;
    float l2_norm = 1.0;

    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    nvtxRangePush("Jacobi Solve Multi-GPU");
    while (l2_norm > tol && iter < iter_max) {
        CUDA_RT_CALL(cudaMemset(l2_norm_d, 0, sizeof(float)));

        launch_jacobi_kernel(a_new, a, l2_norm_d, iy_start, iy_end, nx);
	    CUDA_RT_CALL(cudaDeviceSynchronize());

        CUDA_RT_CALL(cudaMemcpy(l2_norm_h, l2_norm_d, sizeof(float), cudaMemcpyDeviceToHost));

        const int top = rank > 0 ? rank - 1 : (size - 1);
        const int bottom = (rank + 1) % size;

        // Apply periodic boundary conditions

        nvtxRangePush("Halo exchange Memcpy+MPI");
        // First set of halo exchanges
        CUDA_RT_CALL(cudaMemcpy(top_halo_buf, a_new + (iy_start * nx), nx * sizeof(float), 
                                cudaMemcpyDeviceToHost));
        // TODO: Part 1- Implement the first set of halo exchanges using MPI_SendRecv explained 
        // in the Jupyter Notebook. Observe the Memcpy operations above and below this comment
        MPI_CALL(MPI_Sendrecv(/*Fill me*/, nx, MPI_FLOAT, /*Fill me*/, 0,
                              /*Fill me*/, nx, MPI_FLOAT, /*Fill me*/, 0, MPI_COMM_WORLD,
                              MPI_STATUS_IGNORE));
        CUDA_RT_CALL(cudaMemcpy(a_new + (iy_end * nx), bot_halo_buf, nx * sizeof(float), 
                                cudaMemcpyHostToDevice));
        nvtxRangePop();                        

        nvtxRangePush("Halo exchange Memcpy+MPI");
        // Second set of halo exchanges
        // TODO: Part 1- Implement the Memcpy operations and MPI calls for the second set of
        // halo exchanges
        CUDA_RT_CALL(cudaMemcpy(/*Fill me*/, /*Fill me*/, nx * sizeof(float), /*Fill me*/));
        MPI_CALL(MPI_Sendrecv(/*Fill me*/, nx, MPI_FLOAT, /*Fill me*/, 0, 
                                /*Fill me*/, nx, MPI_FLOAT, /*Fill me*/, 0, MPI_COMM_WORLD, 
                                MPI_STATUS_IGNORE));
        CUDA_RT_CALL(cudaMemcpy(/*Fill me*/, /*Fill me*/, nx * sizeof(float), /*Fill me*/));
        nvtxRangePop();                        

        // TODO: Part 1- Reduce the rank-local L2 Norm to a global L2 norm using MPI_Allreduce
        MPI_CALL(MPI_Allreduce(/*Fill me*/, /*Fill me*/, 1, MPI_FLOAT, /*Fill me*/, MPI_COMM_WORLD));
        
        l2_norm = std::sqrt(l2_norm);

        iter++;
        if (0 == rank && (iter % 100) == 0) {
            printf("%5d, %0.6f\n", iter, l2_norm);
        }

        std::swap(a_new, a);
    }
    double stop = MPI_Wtime();
    nvtxRangePop();

    CUDA_RT_CALL(cudaMemcpy(a_h + iy_start_global * nx, a + nx,
                            std::min((ny - iy_start_global) * nx, chunk_size * nx) * sizeof(float),
                            cudaMemcpyDeviceToHost));

    int result_correct = 1;
    if (!skip_single_gpu) {
        for (int iy = iy_start_global; result_correct && (iy < iy_end_global); ++iy) {
            for (int ix = 1; result_correct && (ix < (nx - 1)); ++ix) {
                if (std::fabs(a_ref_h[iy * nx + ix] - a_h[iy * nx + ix]) > tol) {
                    fprintf(stderr,
                            "ERROR on rank %d: a[%d * %d + %d] = %f does not match %f "
                            "(reference)\n",
                            rank, iy, nx, ix, a_h[iy * nx + ix], a_ref_h[iy * nx + ix]);
                    result_correct = 0;
                }
            }
        }
        int global_result_correct = 1;
        MPI_CALL(MPI_Allreduce(&result_correct, &global_result_correct, 1, MPI_INT, MPI_MIN,
                            MPI_COMM_WORLD));
        result_correct = global_result_correct;
    }

    if (rank == 0 && result_correct) {
        printf("Num GPUs: %d.\n", size);
        if (!skip_single_gpu) {
            printf(
                "%dx%d: 1 GPU: %8.4f s, %d GPUs: %8.4f s, speedup: %8.2f, "
                "efficiency: %8.2f \n",
                nx, ny, runtime_serial, size, (stop - start), runtime_serial / (stop - start),
                runtime_serial / (size * (stop - start)) * 100);
        }
        else {
            printf("%dx%d: %d GPUs: %8.4f s \n", nx, ny, size, (stop - start)); 
        }
    }

    CUDA_RT_CALL(cudaFreeHost(l2_norm_h));
    CUDA_RT_CALL(cudaFree(l2_norm_d));

    CUDA_RT_CALL(cudaFree(a_new));
    CUDA_RT_CALL(cudaFree(a));

    CUDA_RT_CALL(cudaFreeHost(a_h));
    CUDA_RT_CALL(cudaFreeHost(a_ref_h));

    MPI_CALL(MPI_Finalize());
    return (result_correct == 1) ? 0 : 1;
}

double single_gpu(const int nx, const int ny, const int iter_max, float* const a_ref_h, bool print) {
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
    launch_initialize_boundaries(a, a_new, PI, 0, nx, ny, ny);
    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());
    nvtxRangePop();

    CUDA_RT_CALL(cudaMalloc(&l2_norm_d, sizeof(float)));
    CUDA_RT_CALL(cudaMallocHost(&l2_norm_h, sizeof(float)));

    CUDA_RT_CALL(cudaDeviceSynchronize());

    if (print) {
        printf("Single GPU jacobi relaxation: %d iterations on %d x %d mesh\n", iter_max, nx, ny);
    }

    int iter = 0;
    float l2_norm = 1.0;

    double start = omp_get_wtime();
    nvtxRangePush("Jacobi Solve Single GPU");
    while (l2_norm > tol && iter < iter_max) {
        CUDA_RT_CALL(cudaMemset(l2_norm_d, 0, sizeof(float)));

        // Compute grid points for this iteration
        launch_jacobi_kernel(a_new, a, l2_norm_d, iy_start, iy_end, nx);
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
        if ((iter % 100) == 0 && print) {
            printf("%5d, %0.6f\n", iter, l2_norm);
        }
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