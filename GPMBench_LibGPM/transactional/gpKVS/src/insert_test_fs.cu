/*
 * Copyright (c) 2015 Kai Zhang (kay21s@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>
#include <chrono>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "gpu_hash_fs.h"
double operation_time = 0, persist_time = 0, memcpy_time = 0, pwrite_time = 0;

//GTX 480 has 14 SM, and M2090 has 16 SM
//#define INSERT_BLOCK 16 defined in gpu_hash.h
#define HASH_BLOCK_ELEM_NUM (BUC_NUM/INSERT_BLOCK)
#define BLOCK_ELEM_NUM (SELEM_NUM/INSERT_BLOCK)


//#define KERNEL 1

int main(int argc, char *argv[])
{
	int SELEM_NUM, THREAD_NUM;
	if (argc != 3) {
		printf("usage: ./run #elem_num #thread_num, now running with 16384\n");
		SELEM_NUM = 16384 * 128;
		THREAD_NUM = 16384 * 2;
	} else {
		SELEM_NUM = atoi(argv[1]);
		THREAD_NUM = atoi(argv[2]);
	}
	printf("elem_num is %d, thread_num is %d\n", SELEM_NUM, THREAD_NUM);

	uint8_t *device_hash_table;
	uint8_t *device_in;
	uint8_t *host_in;

	ielem_t *blk_input_h[INSERT_BLOCK];
	int	blk_elem_num_h[INSERT_BLOCK];
	ielem_t **blk_input_d;
	int *blk_elem_num_d;

	uint8_t *host_hash_table;
	host_hash_table = (uint8_t*)malloc(HT_SIZE);

	double diff;
	int i;

	struct  timespec start, end;
#if defined(KERNEL)
	struct  timespec kernel_start;
#endif
	uint8_t *device_search_in;
	uint8_t *device_search_out;
	uint8_t *host_search_in;
	uint8_t *host_search_out;
	uint8_t *host_search_verify;

	CUDA_SAFE_CALL(cudaMalloc((void **)&(device_hash_table), HT_SIZE));
	CUDA_SAFE_CALL(cudaMemset((void *)device_hash_table, 0, HT_SIZE));

	CUDA_SAFE_CALL(cudaMalloc((void **)&(device_in), SELEM_NUM * sizeof(ielem_t)));
	CUDA_SAFE_CALL(cudaMemset((void *)device_in, 0, SELEM_NUM * sizeof(ielem_t)));
	CUDA_SAFE_CALL(cudaHostAlloc((void **)&(host_in), SELEM_NUM * sizeof(ielem_t), cudaHostAllocDefault));

	CUDA_SAFE_CALL(cudaMalloc((void **)&(blk_input_d), INSERT_BLOCK * sizeof(ielem_t *)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&(blk_elem_num_d), INSERT_BLOCK * sizeof(int)));
	for (i = 0; i < INSERT_BLOCK; i ++) {
		blk_input_h[i] = &(((ielem_t *)device_in)[i*(SELEM_NUM/INSERT_BLOCK)]);
		blk_elem_num_h[i] = SELEM_NUM/INSERT_BLOCK;
	}

	CUDA_SAFE_CALL(cudaMemcpy(blk_input_d, blk_input_h, INSERT_BLOCK * sizeof(void *), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(blk_elem_num_d, blk_elem_num_h, INSERT_BLOCK * sizeof(int), cudaMemcpyHostToDevice));

	// for search
	CUDA_SAFE_CALL(cudaMalloc((void **)&(device_search_in), SELEM_NUM * sizeof(selem_t)));
	CUDA_SAFE_CALL(cudaHostAlloc((void **)&(host_search_in), SELEM_NUM * sizeof(selem_t), cudaHostAllocDefault));
	CUDA_SAFE_CALL(cudaMalloc((void **)&(device_search_out), 2 * SELEM_NUM * sizeof(loc_t)));
	CUDA_SAFE_CALL(cudaHostAlloc((void **)&(host_search_out), 2 * SELEM_NUM * sizeof(loc_t), cudaHostAllocDefault));
	CUDA_SAFE_CALL(cudaHostAlloc((void **)&(host_search_verify), SELEM_NUM * sizeof(loc_t), cudaHostAllocDefault));
	//host_search_verify = (uint8_t *)malloc(SELEM_NUM * sizeof(loc_t));

	// start
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	int has, lower_bond;
	srand(time(NULL));

    double ins_time = 0, search_time = 0, del_time = 0;

    int num_inserts = 25;
    
    int fd = open("/pmem/imkv_fs", O_CREAT | O_RDWR, S_IRWXU | S_IRWXG | S_IRWXO);
    
	for (has = 0; has < num_inserts/*has < 0.1 * HT_SIZE/(sizeof(sign_t) + sizeof(loc_t))*/; has++) {

        printf("%d : Load factor: %f, exisiting number : %d.\n", has, 
                (double)has*SELEM_NUM/(HT_SIZE/(sizeof(sign_t)+sizeof(loc_t))), has*SELEM_NUM);
		/* +++++++++++++++++++++++++++++++++++ INSERT +++++++++++++++++++++++++++++++++ */
		for (i = 0; i < SELEM_NUM; i += 1) {
			lower_bond = (i / BLOCK_ELEM_NUM) * HASH_BLOCK_ELEM_NUM;
			// sig
			((selem_t *)host_search_in)[i].sig
				= ((ielem_t *)host_in)[i].sig = rand();
			// hash
			((selem_t *)host_search_in)[i].hash 
				= ((ielem_t *)host_in)[i].hash 
				= lower_bond + rand() % HASH_BLOCK_ELEM_NUM;
			// loc
			((loc_t *)host_search_verify)[i]
				= ((ielem_t *)host_in)[i].loc = (loc_t)rand(); 
			//printf("%d\n", ((int *)host_search_verify)[i]);
		}
		//for debugging
		for (i = 0; i < SELEM_NUM; i += 1) {
			//printf("%d %d %d\n", ((int *)host_in)[i*3], (i/BLOCK_ELEM_NUM) * BLOCK_ELEM_NUM, 
			//(i/BLOCK_ELEM_NUM) * BLOCK_ELEM_NUM + BLOCK_ELEM_NUM);
			assert(((ielem_t *)host_in)[i].hash < (i/BLOCK_ELEM_NUM) * HASH_BLOCK_ELEM_NUM + HASH_BLOCK_ELEM_NUM);
			assert(((ielem_t *)host_in)[i].hash >= (i/BLOCK_ELEM_NUM) * HASH_BLOCK_ELEM_NUM);
		}


		clock_gettime(CLOCK_MONOTONIC, &start);
		CUDA_SAFE_CALL(cudaMemcpy(device_in, host_in, SELEM_NUM * sizeof(ielem_t), cudaMemcpyHostToDevice));
		gpu_hash_insert((bucket_t *)device_hash_table, 
				(ielem_t **)blk_input_d,
				(int *)blk_elem_num_d, INSERT_BLOCK, SELEM_NUM, fd, 0, operation_time, memcpy_time, pwrite_time, persist_time);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		clock_gettime(CLOCK_MONOTONIC, &end);
		diff = 1000000 * (end.tv_sec-start.tv_sec) + (double)(end.tv_nsec-start.tv_nsec)/1000;
		printf("With Memcpy, the difference is %.2lf us, speed is %.2f Mops\n", 
			(double)diff, (double)(SELEM_NUM) / diff);
		ins_time += diff / 1000.0f;
#if 1
		/* +++++++++++++++++++++++++++++++++++ SEARCH +++++++++++++++++++++++++++++++++ */

		// verify with search
		CUDA_SAFE_CALL(cudaMemcpy(device_search_in, host_search_in, 
				SELEM_NUM * sizeof(selem_t), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemset((void *)device_search_out, 0, 2 * SELEM_NUM * sizeof(loc_t)));
	    gpu_hash_search((selem_t *)device_search_in, (loc_t *)device_search_out, 
			    (bucket_t *)device_hash_table, SELEM_NUM, THREAD_NUM, 128, 0);
	    cudaDeviceSynchronize();
		CUDA_SAFE_CALL(cudaMemcpy(host_search_out, device_search_out, 
				2 * SELEM_NUM * sizeof(loc_t), cudaMemcpyDeviceToHost));

		for (i = 0; i < SELEM_NUM; i ++) {
			if(((loc_t *)host_search_out)[i<<1] != ((loc_t *)host_search_verify)[i] 
				&& ((loc_t *)host_search_out)[(i<<1)+1] != ((loc_t *)host_search_verify)[i]) {
				printf("not found insertion %d : out %lx and %lx, should be : %d\n", i,
						((loc_t *)host_search_out)[i<<1], ((loc_t *)host_search_out)[(i<<1)+1],
						((loc_t *)host_search_verify)[i]);
				/* for debugging
				((int *)host_in)[0] = ((int *)host_in)[i*3];
				((int *)host_in)[1] = ((int *)host_in)[i*3+1];
				((int *)host_in)[2] = ((int *)host_in)[i*3+2];
				CUDA_SAFE_CALL(cudaMemcpy(device_in, host_in, sizeof(ielem_t), cudaMemcpyHostToDevice));
				CUDA_SAFE_CALL(cudaMemset((void *)device_out, 0, SELEM_NUM * sizeof(loc_t)));
				gpu_hash_insert((bucket_t *)device_hash_table, 
					(ielem_t **)blk_input_d, (loc_t **)blk_output_d,
					(int *)blk_elem_num_d, INSERT_BLOCK, 0);
				*/
			}
		}
#endif
	}
	
    printf("\nOperation execution time: %f ms\n", operation_time/1000000.0);
    printf("MemcpyTime\t%f\tms\nPwriteTime\t%f\nPersistTime\t%f\n", memcpy_time/1000000.0f, pwrite_time/1000000.0f, persist_time/1000000.0f);
    printf("Runtime\t%f\tms\n", ins_time);

	return 0;
}
