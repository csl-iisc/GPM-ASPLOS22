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

extern "C" 
{
#include "change-ddio.h"
}
#include <stdio.h>
#include "libgpm.cuh"
#include "bandwidth_analysis.cuh"
#include <chrono>

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include "shfl_integral_image.cuh"
#define RECOVER false

#define TIME_NOW std::chrono::high_resolution_clock::now()
#define time_val std::chrono::duration_cast<std::chrono::microseconds>
double operation_time = 0, persist_time = 0, memcpy_time = 0; 
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

__global__ void shfl_scan_test(int *data, int *pm_partial_sums, int width, int *partial_sums=NULL)
{
    extern __shared__ int sums[];    
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int lane_id = id % warpSize;
    // determine a warp_id within a block
    int warp_id = threadIdx.x / warpSize;
    PMEM_READ_OP(, sizeof(int)) //For the if conditional 
    if(pm_partial_sums != NULL) {
        //if(threadIdx.x == 0)
        //    sums[(blockDim.x + warpSize - 1) / warpSize] = pm_partial_sums[(blockIdx.x + 1) * blockDim.x - 1];
        //__syncthreads();
        // Check if partial sum already calculated for block pre-crash
        PMEM_READ_OP(, sizeof(int)) //For the if conditional 
        if(pm_partial_sums[(blockIdx.x + 1) * blockDim.x - 1] != 0)
            return;
    }

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
        partial_sums[blockIdx.x] = value;
    PMEM_READ_OP(, sizeof(int)) //For the if conditional 
    if(pm_partial_sums != NULL) {
        if(threadIdx.x != blockDim.x - 1)
            gpm_memcpy_nodrain(&pm_partial_sums[id], &value, sizeof(int), cudaMemcpyDeviceToDevice);
        gpm_drain();
        __syncthreads();
        // Persist final partial sums
        if(threadIdx.x == blockDim.x - 1)
            gpm_memcpy(&pm_partial_sums[id], &value, sizeof(int), cudaMemcpyDeviceToDevice);
    }
}

// Uniform add: add partial sums array
__global__ void uniform_add(int *data, int* pm_data, int *partial_sums, int len)
{
    __shared__ int buf;
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);

    PMEM_READ_OP(, sizeof(int)) //For the if conditional 
    if(pm_data[blockIdx.x * blockDim.x] != 0)
        return;

    if (id > len) return;

    if (threadIdx.x == 0)
    {
        if(blockIdx.x == 0)
            buf = 0;
        else {
            PMEM_READ_OP (buf = partial_sums[blockIdx.x - 1], sizeof(int))
        }
    }

    __syncthreads();
    data[id] += buf;
    if(threadIdx.x != 0)
        gpm_memcpy(&pm_data[id], &data[id], sizeof(int), cudaMemcpyDeviceToDevice); 
    __syncthreads();
    if(threadIdx.x == 0) {
        gpm_memcpy(&pm_data[id], &data[id], sizeof(int), cudaMemcpyDeviceToDevice); 
    }
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
    ddio_on();
    int *h_data, *h_partial_sums, *h_result;
    int *d_data, *d_partial_sums;
    const long n_arrays = 1024;
    int blockSize = 1024;
    const long long n_elements = blockSize * blockSize;
    size_t sz = sizeof(int)*n_elements;
    int cuda_device = 0;
    
    int gridSize = n_elements/blockSize;
    int nWarps = blockSize/32;
    int shmem_sz = nWarps * sizeof(int) + sizeof(int);
    int n_partialSums = n_elements/blockSize;
    int partial_sz = n_partialSums*sizeof(int);
    
    size_t len_data = sizeof(int) * n_elements * n_arrays; 
    size_t len_partial_sums = sizeof(int) * n_partialSums * n_arrays; 
    int *pm_data;
    int *pm_partial_sums;
	if(!RECOVER) {
		pm_data = (int*)gpm_map_file("./pm_data.dat", len_data, true);
		cudaMemset(pm_data, 0, len_data);
		pm_partial_sums = (int*) gpm_map_file("./pm_partial_sums.dat", len_partial_sums, true);
		cudaMemset(pm_partial_sums, 0, len_partial_sums);
	} else {
    	auto start_time = TIME_NOW;
		size_t len_data = 0; 
		pm_data = (int*)gpm_map_file("./pm_data.dat", len_data, false); 
		size_t len_partial_sums = 0;
		pm_partial_sums = (int*) gpm_map_file("./pm_partial_sums.dat", len_partial_sums, false);
		double recover_time = (double)time_val(TIME_NOW - start_time).count() / 1000.0;
		printf("Time to recover: %f ms\n", recover_time);
	}
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
    checkCudaErrors(cudaMallocHost((void **)&h_result, sizeof(int)*n_elements));

    //initialize data:
    printf("Computing Simple Sum test\n");
    printf("---------------------------------------------------\n");

    printf("Initialize test data [1, 1, 1...]\n");
    for (long long i=0; i<n_elements; i++)
    {
        h_data[i] = 1;
    }

    printf("Scan summation for %lli elements, %lld partial sums\n",
           n_elements, n_elements/blockSize);

    int p_blockSize = min(n_partialSums, blockSize);
    int p_gridSize = iDivUp(n_partialSums, p_blockSize);
    printf("Partial summing %d elements with %d blocks of size %d\n",
           n_partialSums, p_gridSize, p_blockSize);

    checkCudaErrors(cudaMalloc((void **)&d_data, sz));
    checkCudaErrors(cudaMalloc((void **)&d_partial_sums, partial_sz));
    checkCudaErrors(cudaMallocHost((void **)&h_partial_sums, partial_sz));
    
    auto start1 = TIME_NOW;
    #if defined(NVM_ALLOC_CPU) && !defined(FAKE_NVM) && !defined(GPM_WDP)
        ddio_off(); 
    #endif
    operation_time += (double)time_val(TIME_NOW - start1).count() / 1000.0f;

    //START_BW_MONITOR2("bw_gpm_scan.csv");
    bool success = true;
    for(int iter = 0; iter < n_arrays; ++iter) {
        auto start1 = TIME_NOW;
    	checkCudaErrors(cudaMemcpy(d_data, h_data, sz, cudaMemcpyHostToDevice));
        memcpy_time += (double)time_val(TIME_NOW - start1).count() / 1000.0f;
        checkCudaErrors(cudaMemset(&pm_partial_sums[iter * n_partialSums], 0, partial_sz));
        checkCudaErrors(cudaMemset(d_partial_sums, 0, partial_sz));
        
        start1 = TIME_NOW;
        shfl_scan_test<<<gridSize, blockSize, shmem_sz>>>(d_data, NULL, 32, d_partial_sums);
        shfl_scan_test<<<p_gridSize, p_blockSize, shmem_sz>>>(d_partial_sums, &pm_partial_sums[iter * n_partialSums], 32);
        uniform_add<<<gridSize, blockSize>>>(d_data, &pm_data[iter * n_elements], &pm_partial_sums[iter * n_partialSums], n_elements);
        cudaDeviceSynchronize();
        operation_time += (double)time_val(TIME_NOW - start1).count() / 1000.0f;

        start1 = TIME_NOW;
        checkCudaErrors(cudaMemcpy(h_result, d_data, sz, cudaMemcpyDeviceToHost));
        memcpy_time += (double)time_val(TIME_NOW - start1).count() / 1000.0f;
        //if(!CPUverify(h_data, h_result, n_elements))
        //    success = false;
    }
    //STOP_BW_MONITOR
    OUTPUT_STATS
    
    printf("Successful? %s; ", success ? "True" : "False");
    start1 = TIME_NOW;
    #if defined(NVM_ALLOC_CPU) && !defined(FAKE_NVM) && !defined(GPM_WDP)
        ddio_on(); 
    #endif
    operation_time += (double)time_val(TIME_NOW - start1).count() / 1000.0f; 
    printf("Operation time: %f ms\n", operation_time);
    printf("runtime: %f ms\n", operation_time);
    printf("memcpy_time: %f ms\n", memcpy_time);
    #ifdef GPM_WDP
    start1 = TIME_NOW;
    pmem_mt_persist(pm_data, sz * n_arrays);
    persist_time += (double)time_val(TIME_NOW- start1).count() / 1000.0f; 
    printf("Persist time: %f ms\n", persist_time);
    #endif   
    
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_partial_sums));
    checkCudaErrors(cudaFreeHost(h_data));
    //printf("Operation time: %f ms\n", operation_time);

    return 0;
}

int main(int argc, char *argv[])
{
    shuffle_simple_test(argc, argv);
    return 0;
}
