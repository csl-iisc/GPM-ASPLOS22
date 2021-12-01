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
#include <libpmem.h>
#include "libgpm.cuh"
#include <chrono>

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include "shfl_integral_image.cuh"
#define RECOVER false

#define TIME_NOW std::chrono::high_resolution_clock::now()
#define time_val std::chrono::duration_cast<std::chrono::microseconds>
double operation_time = 0; 
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
    const long n_arrays = 1024;
    int blockSize = 1024;
    const long long n_elements = blockSize * blockSize;
    size_t sz = sizeof(int)*n_elements;
    int nthreads = 32;
    if(argc >= 2)
    	nthreads = atoi(argv[1]);
    else {
    	printf("Need to pass number of CPU threads as commandline argument\n");
    	assert(false);
    }
    int n_partialSums = nthreads;
    int partial_sz = n_partialSums*sizeof(int);
    
    size_t len_data = sizeof(int) * n_elements * n_arrays; 
    size_t len_partial_sums = sizeof(int) * n_partialSums * n_arrays; 
    int *pm_data;
    int *pm_partial_sums;
	pm_data = (int*)gpm_map_file("./pm_data.dat", len_data, true);
	memset(pm_data, 0, len_data);
	pm_partial_sums = (int*) gpm_map_file("./pm_partial_sums.dat", len_partial_sums, true);
	memset(pm_partial_sums, 0, len_partial_sums);
    printf("Starting shfl_scan\n");

	h_data = (int *)malloc(sizeof(int)*n_elements);
	h_result = (int *)malloc(sizeof(int)*n_elements);

    //initialize data:
    printf("Computing Simple Sum test\n");
    printf("---------------------------------------------------\n");

    printf("Initialize test data [1, 1, 1...]\n");
    for (long long i=0; i<n_elements; i++)
        h_data[i] = 1;

    printf("Scan summation for %lli elements, %lld partial sums\n",
           n_elements, n_elements/nthreads);

    h_partial_sums = (int*)malloc(partial_sz);

    bool success = true;
    for(int iter = 0; iter < n_arrays; ++iter) {
    	int *pm_part_sum = &pm_partial_sums[iter * n_partialSums];
    	int *pm_data_sum = &pm_data[iter * n_elements];
        memset(h_partial_sums, 0, partial_sz);
        
        const long long elements_per_thread = (n_elements + nthreads - 1) / nthreads;
        auto start1 = TIME_NOW;
    	#pragma omp parallel for num_threads(nthreads)
        for(int i = 0; i < nthreads; ++i) {
        	// Check if step already done
        	if(pm_part_sum[i] != 0)
        		continue;
        	// Perform partial sum
        	int sum = 0;
        	for(int j = i * elements_per_thread; j < min((i + 1) * elements_per_thread, n_elements); ++j) {
        		sum += h_data[j];
        		h_result[j] = sum;
        	}
        	// Store partial sum
        	h_partial_sums[i] = sum;
        }
        // Have single thread sum the partial sums
        {
        	int sum = 0;
        	for(int j = 0; j < nthreads; ++j) {
        		// If already summed skip
		    	if(pm_part_sum[j] != 0)
		    		continue;
        		sum += h_partial_sums[j];
        		h_partial_sums[j] = sum;
        		pmem_memcpy_nodrain(&pm_part_sum[j], &sum, sizeof(int));
        		pmem_drain();
        	}
        }
    	#pragma omp parallel for num_threads(nthreads)
        for(int i = 1; i < nthreads; ++i) {
        	// Perform partial sum
        	int sum = 0;
        	int start = i * elements_per_thread;
        	int end = min((i + 1) * elements_per_thread, n_elements);
	    	// Check if step already done
	    	if(pm_data_sum[end - 1] != 0)
	    		continue;
        	for(int j = start; j < end; ++j) {
        		h_result[j] += pm_part_sum[i - 1];
        		pmem_memcpy_nodrain(&pm_data_sum[j], &h_result[j], sizeof(int));
        		pmem_drain();
        	}
        }
        operation_time += (double)time_val(TIME_NOW - start1).count() / 1000.0f;
        if(!CPUverify(h_data, h_result, n_elements))
            success = false;
    }
    
    printf("Successful? %s; ", success ? "True" : "False");
    printf("Operation time: %f ms\n", operation_time);
    printf("runtime: %f ms\n", operation_time);
    return 0;
}

int main(int argc, char *argv[])
{
    shuffle_simple_test(argc, argv);
    return 0;
}
