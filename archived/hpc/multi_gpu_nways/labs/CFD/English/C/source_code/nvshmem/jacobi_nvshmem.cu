/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <cub/block/block_reduce.cuh>
#include <nvToolsExt.h>

#define BLOCK_DIM_X 1024
#define BLOCK_DIM_Y 1

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

// convert NVSHMEM_SYMMETRIC_SIZE string to long long unsigned int
long long unsigned int parse_nvshmem_symmetric_size(char *value) {
    long long unsigned int units, size;

    assert(value != NULL);

    if (strchr(value, 'G') != NULL) {
        units=1e9;
    } else if (strchr(value, 'M') != NULL) {
        units=1e6;
    } else if (strchr(value, 'K') != NULL) {
        units=1e3;
    } else {
        units=1;
    }

    assert(atof(value) >= 0);
    size = (long long unsigned int) atof(value) * units;

    return size;
}

constexpr float tol = 1.0e-8;
const float PI = 2.0 * std::asin(1.0);

__global__ void initialize_boundaries(float* __restrict__ const a_new, float* __restrict__ const a,
                                      const float pi, const int offset, const int nx,
                                      const int my_ny, int ny) {
    for (int iy = blockIdx.x * blockDim.x + threadIdx.x; iy < my_ny; iy += blockDim.x * gridDim.x) {
        const float y0 = sin(2.0 * pi * (offset + iy) / (ny - 1));
        a[(iy + 1) * nx + 0] = y0;
        a[(iy + 1) * nx + (nx - 1)] = y0;
        a_new[(iy + 1) * nx + 0] = y0;
        a_new[(iy + 1) * nx + (nx - 1)] = y0;
    }
}

__global__ void jacobi_kernel(float* __restrict__ const a_new, const float* __restrict__ const a,
                              float* __restrict__ const l2_norm, const int iy_start,
                              const int iy_end, const int nx, const int top_pe, const int top_iy,
                              const int bottom_pe, const int bottom_iy) {
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

    /* starting (x, y) coordinate of the block */
    int block_iy = iy - threadIdx.y; /* That is, block_iy = blockIdx.y * blockDim.y + iy_start */
    int block_ix = ix - threadIdx.x; /* That is, block_ix = blockIdx.x * blockDim.x + 1 */

    /* Communicate the boundaries */
    // TODO: Use block-level NVSHMEM put communication API to transfer the halo pointed to by 
    // "a_new + iy_start * nx + block_ix" to "a_new + top_iy * nx + block_ix" in the "top_pe"
    if ((block_iy <= iy_start) && (iy_start < block_iy + blockDim.y)) {
        nvshmemx_/*Fill me*/(/*Fill me*/, /*Fill me*/,
                                   min(blockDim.x, nx - 1 - block_ix), /*Fill me*/);
    }
    // TODO: Use block-level NVSHMEM put communication API to transfer the halo pointed to by 
    // "a_new + (iy_end - 1) * nx + block_ix" to "a_new + bottom_iy * nx + block_ix" in the "bottom_pe"
    if ((block_iy < iy_end) && (iy_end <= block_iy + blockDim.y)) {
        nvshmemx_/*Fill me*/(/*Fill me*/, /*Fill me*/,
                                   min(blockDim.x, nx - 1 - block_ix), /*Fill me*/);
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

double single_gpu(const int nx, const int ny, const int iter_max, float* const a_ref_h,
                  const bool print, int mype);

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

struct l2_norm_buf {
    cudaEvent_t copy_done;
    float* d;
    float* h;
};

int main(int argc, char* argv[]) {
    const int iter_max = get_argval(argv, argv + argc, "-niter", 1000);
    const int nx = get_argval(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval(argv, argv + argc, "-ny", 16384);
    const bool skip_single_gpu = get_arg(argv, argv + argc, "-skip_single_gpu");

    float* a_new;

    float* a_ref_h;
    float* a_h;
    double runtime_serial = 0.0;

    float l2_norms[2];

    int rank = 0, size = 1;
    MPI_CALL(MPI_Init(&argc, &argv));
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));

    int num_devices;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));

    int local_rank = -1, local_size = 1;
    {
        MPI_Comm local_comm;
        MPI_Info info;
        MPI_CALL(MPI_Info_create(&info));
        MPI_CALL(
            MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, info, &local_comm));

        MPI_CALL(MPI_Comm_rank(local_comm, &local_rank));
        MPI_CALL(MPI_Comm_size(local_comm, &local_size));
        if (num_devices < local_size) {
            fprintf(stderr,
                    "ERROR: Number of devices is less numer of PEs \
                    on the node!\n");
            MPI_CALL(MPI_Comm_free(&local_comm));
            MPI_CALL(MPI_Info_free(&info));
            MPI_CALL(MPI_Finalize());
            return -1;
        }

        MPI_CALL(MPI_Comm_free(&local_comm));
        MPI_CALL(MPI_Info_free(&info));
    }
    CUDA_RT_CALL(cudaSetDevice(local_rank));
    CUDA_RT_CALL(cudaFree(0));

    MPI_Comm mpi_comm;
    nvshmemx_init_attr_t attr;

    mpi_comm = MPI_COMM_WORLD;
    attr.mpi_comm = &mpi_comm;
    // Set symmetric heap size for nvshmem based on problem size
    // Its default value in nvshmem is 1 GB which is not sufficient
    // for large mesh sizes
    long long unsigned int mesh_size_per_rank = nx * (((ny - 2) + size - 1) / size + 2);
    long long unsigned int required_symmetric_heap_size =
        2 * mesh_size_per_rank * sizeof(float) *
        1.1;  // Factor 2 is because 2 arrays are allocated - a and a_new
              // 1.1 factor is just for alignment or other usage

    char * value = getenv("NVSHMEM_SYMMETRIC_SIZE");
    if (value) { /* env variable is set */
        long long unsigned int size_env = parse_nvshmem_symmetric_size(value);
        if (size_env < required_symmetric_heap_size) {
            fprintf(stderr, "ERROR: Minimum NVSHMEM_SYMMETRIC_SIZE = %lluB, Current NVSHMEM_SYMMETRIC_SIZE = %s\n", required_symmetric_heap_size, value);
            MPI_CALL(MPI_Finalize());
            return -1;
        }
    } else {
        char symmetric_heap_size_str[100];
        sprintf(symmetric_heap_size_str, "%llu", required_symmetric_heap_size);
        if (!rank)
            printf("Setting environment variable NVSHMEM_SYMMETRIC_SIZE = %llu\n", required_symmetric_heap_size);
        setenv("NVSHMEM_SYMMETRIC_SIZE", symmetric_heap_size_str, 1);
    }
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    int npes = nvshmem_n_pes();
    int mype = nvshmem_my_pe();

    nvshmem_barrier_all();

    bool result_correct = true;
    float* a;

    cudaStream_t compute_stream;
    cudaStream_t reset_l2_norm_stream;
    cudaEvent_t compute_done[2];
    cudaEvent_t reset_l2_norm_done[2];

    l2_norm_buf l2_norm_bufs[2];

    CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * sizeof(float)));
    CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * sizeof(float)));
    if (!skip_single_gpu){
        runtime_serial = single_gpu(nx, ny, iter_max, a_ref_h, (0 == mype), mype);
    }
    
    nvshmem_barrier_all();
    // ny - 2 rows are distributed amongst `size` ranks in such a way
    // that each rank gets either (ny - 2) / size or (ny - 2) / size + 1 rows.
    // This optimizes load balancing when (ny - 2) % size != 0
    int chunk_size;
    int chunk_size_low = (ny - 2) / npes;
    int chunk_size_high = chunk_size_low + 1;
    // To calculate the number of ranks that need to compute an extra row,
    // the following formula is derived from this equation:
    // num_ranks_low * chunk_size_low + (size - num_ranks_low) * (chunk_size_low + 1) = ny - 2
    int num_ranks_low = npes * chunk_size_low + npes -
                        (ny - 2);  // Number of ranks with chunk_size = chunk_size_low
    if (mype < num_ranks_low)
        chunk_size = chunk_size_low;
    else
        chunk_size = chunk_size_high;

    a = (float*)nvshmem_malloc(
        nx * (chunk_size_high + 2) *
        sizeof(float));  // Using chunk_size_high so that it is same across all PEs
    a_new = (float*)nvshmem_malloc(nx * (chunk_size_high + 2) * sizeof(float));

    cudaMemset(a, 0, nx * (chunk_size + 2) * sizeof(float));
    cudaMemset(a_new, 0, nx * (chunk_size + 2) * sizeof(float));

    // Calculate local domain boundaries
    int iy_start_global;  // My start index in the global array
    if (mype < num_ranks_low) {
        iy_start_global = mype * chunk_size_low + 1;
    } else {
        iy_start_global =
            num_ranks_low * chunk_size_low + (mype - num_ranks_low) * chunk_size_high + 1;
    }
    int iy_end_global = iy_start_global + chunk_size - 1;  // My last index in the global array
    // do not process boundaries
    iy_end_global = std::min(iy_end_global, ny - 4);

    int iy_start = 1;
    int iy_end = (iy_end_global - iy_start_global + 1) + iy_start;

    // calculate boundary indices for top and bottom boundaries
    int top_pe = mype > 0 ? mype - 1 : (npes - 1);
    int bottom_pe = (mype + 1) % npes;

    int iy_end_top = (top_pe < num_ranks_low) ? chunk_size_low + 1 : chunk_size_high + 1;
    int iy_start_bottom = 0;

    // Set diriclet boundary conditions on left and right boundary
    initialize_boundaries<<<(ny / npes) / 128 + 1, 128>>>(a, a_new, PI, iy_start_global - 1, nx,
                                                          chunk_size, ny - 2);
    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());

    CUDA_RT_CALL(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    CUDA_RT_CALL(cudaStreamCreate(&reset_l2_norm_stream));
    CUDA_RT_CALL(cudaEventCreate(&compute_done[0]));
    CUDA_RT_CALL(cudaEventCreate(&compute_done[1]));
    CUDA_RT_CALL(cudaEventCreate(&reset_l2_norm_done[0]));
    CUDA_RT_CALL(cudaEventCreate(&reset_l2_norm_done[1]));

    for (int i = 0; i < 2; ++i) {
        CUDA_RT_CALL(cudaEventCreate(&l2_norm_bufs[i].copy_done));
        CUDA_RT_CALL(cudaMalloc(&l2_norm_bufs[i].d, sizeof(float)));
        CUDA_RT_CALL(cudaMemset(l2_norm_bufs[i].d, 0, sizeof(float)));
        CUDA_RT_CALL(cudaMallocHost(&l2_norm_bufs[i].h, sizeof(float)));
        *(l2_norm_bufs[i].h) = 1.0;
    }

    nvshmemx_barrier_all_on_stream(compute_stream);
    MPI_CALL(MPI_Allreduce(l2_norm_bufs[0].h, &l2_norms[0], 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
    MPI_CALL(MPI_Allreduce(l2_norm_bufs[1].h, &l2_norms[1], 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
    CUDA_RT_CALL(cudaDeviceSynchronize());

    if (!mype) {
        printf("Jacobi relaxation: %d iterations on %d x %d mesh\n", iter_max, nx, ny);
    }

    dim3 dim_grid((nx + BLOCK_DIM_X-1) / BLOCK_DIM_X, (chunk_size + BLOCK_DIM_Y-1) / BLOCK_DIM_Y, 1);
    dim3 dim_block(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

    int iter = 0;
    if (!mype) {
        for (int i = 0; i < 2; ++i) {
            l2_norms[i] = 1.0;
        }
    }

    nvshmem_barrier_all();

    double start = MPI_Wtime();
    bool l2_norm_greater_than_tol = true;

    nvtxRangePush("Jacobi Solve Multi-GPU");
    while (l2_norm_greater_than_tol && iter < iter_max) {
        // on new iteration: old current vars are now previous vars, old
        // previous vars are no longer needed
        int prev = iter % 2;
        int curr = (iter + 1) % 2;

        // TODO: Use "cudaStreamWaitEvent" on "compute_stream" to wait for "reset_l2_norm_done"
        // event to complete for "curr" iteration
        CUDA_RT_CALL(cudaStreamWaitEvent(/*Fill me*/, /*Fill me*/, 0));
        jacobi_kernel<<<dim_grid, dim_block, 0, compute_stream>>>(
                a_new, a, l2_norm_bufs[curr].d, iy_start, iy_end, nx, top_pe, iy_end_top, bottom_pe,
                iy_start_bottom);

        // TODO: Put a barrier at the "compute_stream" stream level
        nvshmemx_/*Fill me*/(/*Fill me*/);

        // perform L2 norm calculation
        // as soon as computation is complete -> D2H-copy L2 norm
        CUDA_RT_CALL(cudaMemcpyAsync(l2_norm_bufs[curr].h, l2_norm_bufs[curr].d, sizeof(float),
                                        cudaMemcpyDeviceToHost, compute_stream));
        // TODO: Record the event "l2_norm_bufs[curr].copy_done" for "compute_stream"
        CUDA_RT_CALL(cudaEventRecord(/*Fill me*/, /*Fill me*/));

        // ensure previous D2H-copy is completed before using the data for
        // calculation
        CUDA_RT_CALL(cudaEventSynchronize(l2_norm_bufs[prev].copy_done));

        MPI_CALL(MPI_Allreduce(l2_norm_bufs[prev].h, &l2_norms[prev], 1, MPI_FLOAT, MPI_SUM,
                                MPI_COMM_WORLD));

        l2_norms[prev] = std::sqrt(l2_norms[prev]);
        l2_norm_greater_than_tol = (l2_norms[prev] > tol);

        iter++;
        if ((iter % 100) == 0) {
            if (!mype) printf("%5d, %0.6f\n", iter, l2_norms[prev]);
        }

        // reset everything for next iteration
        l2_norms[prev] = 0.0;
        *(l2_norm_bufs[prev].h) = 0.0;
        CUDA_RT_CALL(cudaMemcpyAsync(l2_norm_bufs[prev].d, l2_norm_bufs[prev].h, sizeof(float),
                                        cudaMemcpyHostToDevice, reset_l2_norm_stream));
        // TODO: Record the L2 norm reset in "reset_l2_norm_done[prev]" for "reset_l2_norm_stream"
        CUDA_RT_CALL(cudaEventRecord(/*Fill me*/, /*Fill me*/));

        std::swap(a_new, a);
    }
    CUDA_RT_CALL(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    double stop = MPI_Wtime();
    nvtxRangePop();

    nvshmem_barrier_all();

    CUDA_RT_CALL(cudaMemcpy(a_h + iy_start_global * nx, a + nx,
                            std::min(ny - 2 - iy_start_global, chunk_size) * nx * sizeof(float),
                            cudaMemcpyDeviceToHost));

    result_correct = true;
    if (!skip_single_gpu) {
        for (int iy = iy_start_global; result_correct && (iy < iy_end_global); ++iy) {
            for (int ix = 1; result_correct && (ix < (nx - 1)); ++ix) {
                if (std::fabs(a_ref_h[iy * nx + ix] - a_h[iy * nx + ix]) > tol) {
                    fprintf(stderr,
                            "ERROR on rank %d: a[%d * %d + %d] = %f does not match %f "
                            "(reference)\n",
                            rank, iy, nx, ix, a_h[iy * nx + ix], a_ref_h[iy * nx + ix]);
                    result_correct = false;
                }
            }
        }
        int global_result_correct = 1;
        MPI_CALL(MPI_Allreduce(&result_correct, &global_result_correct, 1, MPI_INT, MPI_MIN,
                                MPI_COMM_WORLD));
        result_correct = global_result_correct;
    }

    if (!mype && result_correct) {
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

    for (int i = 0; i < 2; ++i) {
        CUDA_RT_CALL(cudaFreeHost(l2_norm_bufs[i].h));
        CUDA_RT_CALL(cudaFree(l2_norm_bufs[i].d));
        CUDA_RT_CALL(cudaEventDestroy(l2_norm_bufs[i].copy_done));
    }

    nvshmem_free(a);
    nvshmem_free(a_new);

    CUDA_RT_CALL(cudaEventDestroy(reset_l2_norm_done[1]));
    CUDA_RT_CALL(cudaEventDestroy(reset_l2_norm_done[0]));
    CUDA_RT_CALL(cudaEventDestroy(compute_done[1]));
    CUDA_RT_CALL(cudaEventDestroy(compute_done[0]));
    CUDA_RT_CALL(cudaStreamDestroy(reset_l2_norm_stream));
    CUDA_RT_CALL(cudaStreamDestroy(compute_stream));

    CUDA_RT_CALL(cudaFreeHost(a_h));
    CUDA_RT_CALL(cudaFreeHost(a_ref_h));

    nvshmem_finalize();
    MPI_CALL(MPI_Finalize());

    return (result_correct == 1) ? 0 : 1;
}

double single_gpu(const int nx, const int ny, const int iter_max, float* const a_ref_h,
                  const bool print, int mype) {
    float* a;
    float* a_new;
    float* l2_norm_d;
    float* l2_norm_h;

    int iy_start = 1;
    int iy_end = ny - 3;

    CUDA_RT_CALL(cudaMalloc((void**)&a, nx * ny * sizeof(float)));
    CUDA_RT_CALL(cudaMalloc((void**)&a_new, nx * ny * sizeof(float)));

    CUDA_RT_CALL(cudaMemset(a, 0, nx * ny * sizeof(float)));
    CUDA_RT_CALL(cudaMemset(a_new, 0, nx * ny * sizeof(float)));

    // Set diriclet boundary conditions on left and right boarder
    initialize_boundaries<<<ny / 128 + 1, 128>>>(a, a_new, PI, 0, nx, ny - 2, ny - 2);

    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());

    CUDA_RT_CALL(cudaMalloc(&l2_norm_d, sizeof(float)));
    CUDA_RT_CALL(cudaMallocHost(&l2_norm_h, sizeof(float)));

    CUDA_RT_CALL(cudaDeviceSynchronize());

    if (print)
        printf("Single GPU jacobi relaxation: %d iterations on %d x %d mesh\n", iter_max, nx, ny);

    dim3 dim_block(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    dim3 dim_grid((nx + BLOCK_DIM_X-1) / BLOCK_DIM_X, ((ny - 2) + BLOCK_DIM_Y-1) / BLOCK_DIM_Y, 1);

    int iter = 0;
    float l2_norm = 1.0;

    CUDA_RT_CALL(cudaDeviceSynchronize());
    double start = MPI_Wtime();
    nvtxRangePush("Jacobi Solve Single GPU");

    while (l2_norm > tol && iter < iter_max) {
        CUDA_RT_CALL(cudaMemsetAsync(l2_norm_d, 0, sizeof(float)));

        jacobi_kernel<<<dim_grid, dim_block>>>(a_new, a, l2_norm_d, iy_start, iy_end, nx, 
                                                mype, iy_end + 1, mype, (iy_start - 1));
        
        iter++;
        if (print && ((iter % 100) == 0)) {
            CUDA_RT_CALL(cudaMemcpy(l2_norm_h, l2_norm_d, sizeof(float), cudaMemcpyDeviceToHost));
            l2_norm = *l2_norm_h;
            l2_norm = std::sqrt(l2_norm);
            if (print && (iter % 100) == 0) printf("%5d, %0.6f\n", iter, l2_norm);
        }

        std::swap(a_new, a);
    }
    CUDA_RT_CALL(cudaDeviceSynchronize());
    nvtxRangePop();
    double stop = MPI_Wtime();

    CUDA_RT_CALL(cudaMemcpy(a_ref_h, a, nx * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(cudaFreeHost(l2_norm_h));
    CUDA_RT_CALL(cudaFree(l2_norm_d));

    CUDA_RT_CALL(cudaFree(a_new));
    CUDA_RT_CALL(cudaFree(a));
    return (stop - start);
}
