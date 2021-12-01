/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Shuffle intrinsics CUDA Sample
// This sample demonstrates the use of the shuffle intrinsic
// First, a simple example of a prefix sum using the shuffle to
// perform a scan operation is provided.
// Secondly, a more involved example of computing an integral image
// using the shuffle intrinsic is provided, where the shuffle
// scan operation and shuffle xor operations are used

#include <stdio.h>

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include "shfl_integral_image.cuh"
#include "bandwidth_analysis.cuh"
#include "libgpm.cuh"

#include <unistd.h>
#include <thread>
#include <assert.h>
#include <libpmem.h>
#include <chrono>

#define TIME_NOW std::chrono::high_resolution_clock::now()
double operation_time = 0, memcpy_time = 0, persist_time = 0;
long long nvm_writes = 0; 

// Scan using shfl - takes log2(n) steps
// This function demonstrates basic use of the shuffle intrinsic, __shfl_up,
// to perform a scan operation across a block.
// First, it performs a scan (prefix sum in this case) inside a warp
// Then to continue the scan operation across the block,
// each warp's sum is placed into shared memory.  A single warp
// then performs a shuffle scan on that shared memory.  The results
// are then uniformly added to each warp's threads.
// This pyramid type approach is continued by placing each block's
// final sum in global memory and prefix summing that via another kernel call, then
// uniformly adding across the input data via the uniform_add<<<>>> kernel.

__global__ void shfl_scan_test(int *data, int width, int *partial_sums=NULL)
{
    extern __shared__ int sums[];
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int lane_id = id % warpSize;
    // determine a warp_id within a block
    int warp_id = threadIdx.x / warpSize;

    // Below is the basic structure of using a shfl instruction
    // for a scan.
    // Record "value" as a variable - we accumulate it along the way
    int value = data[id];

    // Now accumulate in log steps up the chain
    // compute sums, with another thread's value who is
    // distance delta away (i).  Note
    // those threads where the thread 'i' away would have
    // been out of bounds of the warp are unaffected.  This
    // creates the scan sum.

#pragma unroll
    for (int i=1; i<=width; i*=2)
    {
        unsigned int mask = 0xffffffff;
        int n = __shfl_up_sync(mask, value, i, width);

        if (lane_id >= i) value += n;
    }

    // value now holds the scan value for the individual thread
    // next sum the largest values for each warp

    // write the sum of the warp to smem
    if (threadIdx.x % warpSize == warpSize-1)
    {
        sums[warp_id] = value;
    }

    __syncthreads();

    //
    // scan sum the warp sums
    // the same shfl scan operation, but performed on warp sums
    //
    if (warp_id == 0 && lane_id < (blockDim.x / warpSize))
    {
        int warp_sum = sums[lane_id];

        int mask = (1 << (blockDim.x / warpSize)) - 1;
        for (int i=1; i<=(blockDim.x / warpSize); i*=2)
        {
            int n = __shfl_up_sync(mask, warp_sum, i, (blockDim.x / warpSize));

            if (lane_id >= i) warp_sum += n;
        }

        sums[lane_id] = warp_sum;
    }

    __syncthreads();

    // perform a uniform add across warps in the block
    // read neighbouring warp's sum and add it to threads value
    int blockSum = 0;

    if (warp_id > 0)
    {
        blockSum = sums[warp_id-1];
    }

    value += blockSum;

    // Now write out our result
    data[id] = value;

    // last thread has sum, write write out the block's sum
    if (partial_sums != NULL && threadIdx.x == blockDim.x-1)
    {
        partial_sums[blockIdx.x] = value;
    }
}

// Uniform add: add partial sums array
__global__ void uniform_add(int *data, int *partial_sums, int len)
{
    __shared__ int buf;
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);

    if (id > len) return;

    if (threadIdx.x == 0)
    {
        buf = partial_sums[blockIdx.x];
    }

    __syncthreads();
    data[id] += buf;
}

static unsigned int iDivUp(unsigned int dividend, unsigned int divisor)
{
    return ((dividend % divisor) == 0) ?
           (dividend / divisor) :
           (dividend / divisor + 1);
}


// This function verifies the shuffle scan result, for the simple
// prefix sum case.
bool CPUverify(int *h_data, int *h_result, int n_elements)
{
    // cpu verify
    for (int i=0; i<n_elements-1; i++)
    {
        h_data[i+1] = h_data[i] + h_data[i+1];
    }

    int diff = 0;

    for (int i=0 ; i<n_elements; i++)
    {
        diff += h_data[i]-h_result[i];
    }

    if(diff != 0)
        printf("CPU verify result diff (GPUvsCPU) = %d\n", diff);
    bool bTestResult = false;

    if (diff == 0) bTestResult = true;

    StopWatchInterface *hTimer = NULL;
    sdkCreateTimer(&hTimer);
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    for (int j=0; j<100; j++)
        for (int i=0; i<n_elements-1; i++)
        {
            h_data[i+1] = h_data[i] + h_data[i+1];
        }

    sdkStopTimer(&hTimer);
    double cput= sdkGetTimerValue(&hTimer);
    //printf("CPU sum (naive) took %f ms\n", cput/100);
    return bTestResult;
}


// this verifies the row scan result for synthetic data of all 1's
unsigned int verifyDataRowSums(unsigned int *h_image, int w, int h)
{
    unsigned int diff = 0;

    for (int j=0; j<h; j++)
    {
        for (int i=0; i<w; i++)
        {
            int gold = i+1;
            diff += abs((int)gold-(int)h_image[j*w + i]);
        }
    }

    return diff;
}

bool shuffle_simple_test(int argc, char **argv)
{
    int *h_data, *h_partial_sums, *h_result;
    int *d_data, *d_partial_sums;
    const long long n_elements = 1024 * 1024;
    const long n_arrays = 1024;
    size_t sz = sizeof(int)*n_elements;
    int cuda_device = 0;

    printf("Starting shfl_scan\n");

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    cuda_device = findCudaDevice(argc, (const char **)argv);

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDevice(&cuda_device));

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

    printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // __shfl intrinsic needs SM 3.0 or higher
    if (deviceProp.major < 3)
    {
        printf("> __shfl() intrinsic requires device SM 3.0+\n");
        printf("> Waiving test.\n");
        exit(EXIT_WAIVED);
    }

    checkCudaErrors(cudaMallocHost((void **)&h_data, sizeof(int)*n_elements));
    h_result = (int*)malloc(sizeof(int)*n_elements);

    //initialize data:
    printf("Computing Simple Sum test\n");
    printf("---------------------------------------------------\n");

    printf("Initialize test data [1, 1, 1...]\n");
    for (long long i=0; i<n_elements; i++)
    {
        h_data[i] = 1;
    }
    
    int blockSize = 1024;
    int gridSize = n_elements/blockSize;
    int nWarps = blockSize/32;
    int shmem_sz = nWarps * sizeof(int);
    int n_partialSums = n_elements/blockSize;
    int partial_sz = n_partialSums*sizeof(int);

    printf("Scan summation for %lli elements, %lld partial sums\n",
           n_elements, n_elements/blockSize);

    int p_blockSize = min(n_partialSums, blockSize);
    int p_gridSize = iDivUp(n_partialSums, p_blockSize);
    printf("Partial summing %d elements with %d blocks of size %d\n",
           n_partialSums, p_gridSize, p_blockSize);
    
    size_t tot_size = sz * n_arrays;
    size_t mapped_len;
    int is_pmemp; 
    int *pm_result = (int*) pmem_map_file("/pmem/pm_scan_result", tot_size, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmemp); 

    checkCudaErrors(cudaMalloc((void **)&d_data, sz));
    checkCudaErrors(cudaMalloc((void **)&d_partial_sums, partial_sz));
    checkCudaErrors(cudaMallocHost((void **)&h_partial_sums, partial_sz));

    //START_BW_MONITOR2("bw_mm_scan.csv");
    bool success = true;
    for(int iter = 0; iter < n_arrays; ++iter) {
        checkCudaErrors(cudaMemcpy(d_data, h_data, sz, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemset(d_partial_sums, 0, partial_sz));

        auto start = TIME_NOW; 
        shfl_scan_test<<<gridSize,blockSize, shmem_sz>>>(d_data, 32, d_partial_sums);
        shfl_scan_test<<<p_gridSize,p_blockSize, shmem_sz>>>(d_partial_sums,32);
        uniform_add<<<gridSize-1, blockSize>>>(d_data+blockSize, d_partial_sums, n_elements);
        cudaDeviceSynchronize(); 
        operation_time += (TIME_NOW - start).count(); 

        start = TIME_NOW;
        checkCudaErrors(cudaMemcpy(&pm_result[iter * n_elements], d_data, sz, cudaMemcpyDeviceToHost));
        memcpy_time += (TIME_NOW - start).count(); 
        
        start = TIME_NOW;
        if (is_pmemp) {
            pmem_mt_persist(&pm_result[iter * n_elements], sz);
            nvm_writes += sz; 
        } else {
            pmem_msync(&pm_result[iter * n_elements], sz);
       
        }
        persist_time += (TIME_NOW - start).count();
    
        checkCudaErrors(cudaMemcpy(h_result, d_data, sz, cudaMemcpyDeviceToHost));
        if(!CPUverify(h_data, h_result, n_elements))
            success = false;
    }
    
    //STOP_BW_MONITOR
    printf("Successful? %s; ", success ? "True" : "False");
    checkCudaErrors(cudaFreeHost(h_data));
    checkCudaErrors(cudaFreeHost(h_partial_sums));
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_partial_sums));
    
    printf("\nOperation time: %f ms\n", operation_time/1000000.0);
    printf("memcpy time: %f \t persist time: %f \t\n", memcpy_time/1000000.0f, persist_time/1000000.0f);
    printf("\nruntime: %f ms\n", (operation_time + memcpy_time + persist_time)/1000000.0);
    printf("Tot writes: %lli\n", nvm_writes); 
    return 0;
}

int main(int argc, char *argv[])
{
    shuffle_simple_test(argc, argv);
    return 0;
}
