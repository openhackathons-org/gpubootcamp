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
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <mpi.h>
#include <omp.h>
#include <nccl.h>

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

#define NCCL_CALL(call)                                                                     \
    {                                                                                       \
        ncclResult_t  ncclStatus = call;                                                    \
        if (ncclSuccess != ncclStatus)                                                      \
            fprintf(stderr,                                                                 \
                    "ERROR: NCCL call \"%s\" in line %d of file %s failed "                 \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, ncclGetErrorString(ncclStatus), ncclStatus); \
    }

constexpr float tol = 1.0e-8;

const float PI = 2.0 * std::asin(1.0);

void launch_jacobi_kernel(float*  a_new, const float*  a, float*  l2_norm, const int iy_start,
                              const int iy_end, const int nx, cudaStream_t stream);

void launch_initialize_boundaries(float*  a_new, float*  a, const float pi, const int offset, 
                                    const int nx, const int my_ny, const int ny);

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

    ncclUniqueId nccl_uid;
    if (rank == 0) NCCL_CALL(ncclGetUniqueId(&nccl_uid));
    MPI_CALL(MPI_Bcast(&nccl_uid, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));

    const int iter_max = get_argval(argv, argv + argc, "-niter", 1000);
    const int nx = get_argval(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval(argv, argv + argc, "-ny", 16384);
    const bool skip_single_gpu = get_arg(argv, argv + argc, "-skip_single_gpu");

    int local_rank = -1;
    {
        MPI_Comm local_comm;
        MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                     &local_comm));

        MPI_CALL(MPI_Comm_rank(local_comm, &local_rank));

        MPI_CALL(MPI_Comm_free(&local_comm));
    }

    CUDA_RT_CALL(cudaSetDevice(local_rank));
    CUDA_RT_CALL(cudaFree(0));

    ncclComm_t nccl_comm;
    NCCL_CALL(ncclCommInitRank(&nccl_comm, size, nccl_uid, rank));


    int nccl_version = 0;
    NCCL_CALL(ncclGetVersion(&nccl_version));
    if ( nccl_version < 2800 ) {
        fprintf(stderr,"ERROR NCCL 2.8 or newer is required.\n");
        NCCL_CALL(ncclCommDestroy(nccl_comm));
        MPI_CALL(MPI_Finalize());
        return 1;
    }

    float* a_ref_h;
    CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * sizeof(float)));
    float* a_h;
    CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * sizeof(float)));
    
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
    float* l2_global_norm_d;
    CUDA_RT_CALL(cudaMalloc(&l2_global_norm_d, sizeof(float)));
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
        CUDA_RT_CALL(cudaMemsetAsync(l2_norm_d, 0, sizeof(float)));

        launch_jacobi_kernel(a_new, a, l2_norm_d, iy_start, iy_end, nx, 0);

        const int top = rank > 0 ? rank - 1 : (size - 1);
        const int bottom = (rank + 1) % size;

        // TODO: Reduce the device-local L2 norm, "l2_norm_d" to the global L2 norm on each device,
        // "l2_global_norm_d", using ncclAllReduce() function. Use "ncclSum" as the reduction operation.
        // Make sure to encapsulate this funciton call within NCCL group calls.
        // Use "0" in the stream parameter function argument.
        NCCL_CALL(ncclGroupStart());
        NCCL_CALL(ncclAllReduce(l2_norm_d, l2_global_norm_d, 1, ncclFloat, ncclSum, nccl_comm, 
                                    0));
        NCCL_CALL(ncclGroupEnd());

        // TODO: Transfer the global L2 norm from each device to the host using cudaMemcpyAsync
        CUDA_RT_CALL(cudaMemcpyAsync(l2_norm_h, l2_global_norm_d, sizeof(float), cudaMemcpyDeviceToHost));

        // Apply periodic boundary conditions
        NCCL_CALL(ncclGroupStart());
        
        //TODO: Perform the first set of halo exchanges by:
        // 1. Receiving the top halo from the "top" neighbour into the "a_new" device memory array location. 
        // 2. Sending current device's bottom halo to "bottom" neighbour from the "a_new + (iy_end - 1) * nx"
        //    device memory array location.
        // Use "0" in the stream parameter function argument.
        NCCL_CALL(ncclRecv(a_new,                     nx, ncclFloat, top,    nccl_comm, 0));
        NCCL_CALL(ncclSend(a_new + (iy_end - 1) * nx, nx, ncclFloat, bottom, nccl_comm, 0));

        //TODO: Perform the second set of halo exchanges by:
        // 1. Receiving the bottom halo from the "bottom" neighbour into the "a_new + (iy_end * nx)" 
        //    device memory array location. 
        // 2. Sending current device's top halo to "top" neighbour from the "a_new + iy_start * nx"
        //    device memory array location.
        // Use "0" in the stream parameter function argument.
        NCCL_CALL(ncclRecv(a_new + (iy_end * nx),     nx, ncclFloat, bottom, nccl_comm, 0));
        NCCL_CALL(ncclSend(a_new + iy_start * nx,     nx, ncclFloat, top,    nccl_comm, 0));

        NCCL_CALL(ncclGroupEnd());

        // TODO: Synchronize the device before computing the global L2 norm on host for printing
        CUDA_RT_CALL(cudaDeviceSynchronize());

        l2_norm = *l2_norm_h;
        l2_norm = std::sqrt(l2_norm);

        iter++;
        if (0 == rank && (iter % 100) == 0) {
            printf("%5d, %0.6f\n", iter, l2_norm);
        }

        std::swap(a_new, a);
    }
    CUDA_RT_CALL(cudaDeviceSynchronize());
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

    NCCL_CALL(ncclCommDestroy(nccl_comm));

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
        launch_jacobi_kernel(a_new, a, l2_norm_d, iy_start, iy_end, nx, 0);
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
