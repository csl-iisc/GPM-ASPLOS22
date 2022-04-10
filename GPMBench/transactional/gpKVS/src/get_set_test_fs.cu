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
#define EMULATE_NVM_BW
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>
#include <chrono>
#include <byteswap.h>
#include <map>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "gpu_hash_fs.h"
#include "zipf.h"

#define HASH_BLOCK_ELEM_NUM (BUC_NUM/INSERT_BLOCK)
#define BLOCK_ELEM_NUM (SELEM_NUM/INSERT_BLOCK)
#define LOAD_FACTOR 1 / 8
#define PRELOAD_CNT (uint32_t)(((1 << 30)/8) * LOAD_FACTOR)
#define TOTAL_CNT (((uint32_t)1 << 31) - 1)
#define ZIPF_THETA 0.99
double persist_time = 0, operation_time = 0, memcpy_time = 0, pwrite_time = 0;

#define TIME_NOW std::chrono::high_resolution_clock::now()
#define time_diff(a, b) std::chrono::duration_cast<std::chrono::microseconds>(a - b).count()

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

	struct zipf_gen_state zipf_state;
	mehcached_zipf_init(&zipf_state, (uint64_t)PRELOAD_CNT - 2, (double)ZIPF_THETA, (uint64_t)21);

    uint8_t *device_hash_table;
    uint8_t *device_in;
    uint8_t *host_in;

    ielem_t *blk_input_h[INSERT_BLOCK];
    int	blk_elem_num_h[INSERT_BLOCK];
    ielem_t **blk_input_d;
    int *blk_elem_num_d;

    int i;
    std::map<selem_t, loc_t> cpu_map;

    uint8_t *device_search_in;
    uint8_t *device_search_out;
    uint8_t *host_search_in;
    uint8_t *host_search_out;
    uint8_t *host_search_verify;

	CUDA_SAFE_CALL(cudaMalloc((void **)&(device_hash_table), HT_SIZE));
	CUDA_SAFE_CALL(cudaMemset((void *)device_hash_table, 0, HT_SIZE));

	// Allocate memory for preloading keys into KVS
    CUDA_SAFE_CALL(cudaMalloc((void **)&(device_in), PRELOAD_CNT * sizeof(ielem_t)));
    CUDA_SAFE_CALL(cudaMemset((void *)device_in, 0, PRELOAD_CNT * sizeof(ielem_t)));
    CUDA_SAFE_CALL(cudaHostAlloc((void **)&(host_in), PRELOAD_CNT * sizeof(ielem_t), cudaHostAllocDefault));

    CUDA_SAFE_CALL(cudaMalloc((void **)&(blk_input_d), INSERT_BLOCK * sizeof(ielem_t *)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&(blk_elem_num_d), INSERT_BLOCK * sizeof(int)));
    for (i = 0; i < INSERT_BLOCK; i ++) {
        blk_input_h[i] = &(((ielem_t *)device_in)[i*((int)PRELOAD_CNT/INSERT_BLOCK)]);
        blk_elem_num_h[i] = 0;
    }
    // for search
    CUDA_SAFE_CALL(cudaMalloc((void **)&(device_search_in), PRELOAD_CNT * sizeof(selem_t)));
    CUDA_SAFE_CALL(cudaHostAlloc((void **)&(host_search_in), PRELOAD_CNT * sizeof(selem_t), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaMalloc((void **)&(device_search_out), 2 * PRELOAD_CNT * sizeof(loc_t)));
    CUDA_SAFE_CALL(cudaHostAlloc((void **)&(host_search_out), 2 * PRELOAD_CNT * sizeof(loc_t), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc((void **)&(host_search_verify), PRELOAD_CNT * sizeof(loc_t), cudaHostAllocDefault));
	
	// Generate keys
	printf("Generate %d keys\n", PRELOAD_CNT);
	int num_keys = PRELOAD_CNT / INSERT_BLOCK;
	for(int i = 0; i < PRELOAD_CNT; ++i) {
		int blk = (i + 1) % INSERT_BLOCK;
		int index = num_keys * blk + blk_elem_num_h[blk];
        // sig
        ((ielem_t *)host_in)[index].sig = 
        	((selem_t *)host_search_in)[index].sig =
        	(i + 1);
        // hash
        ((ielem_t *)host_in)[index].hash =
        	((selem_t *)host_search_in)[index].hash =
        	(i + 1);
        // loc
        ((ielem_t *)host_in)[index].loc = (loc_t)rand();
        cpu_map[selem_t(i+1, i+1)] = ((ielem_t *)host_in)[index].loc;
		blk_elem_num_h[blk]++;
	}

    CUDA_SAFE_CALL(cudaMemcpy(blk_input_d, blk_input_h, INSERT_BLOCK * sizeof(void *), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(blk_elem_num_d, blk_elem_num_h, INSERT_BLOCK * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(device_in, host_in, PRELOAD_CNT * sizeof(ielem_t), cudaMemcpyHostToDevice));
    
    int fd = open("/pmem/imkv_fs", O_CREAT | O_RDWR, S_IRWXU | S_IRWXG | S_IRWXO);
	
	// Insert preload keys
	printf("Preload %d keys\n", PRELOAD_CNT);
    double ins_time = 0, search_time = 0, del_time = 0;
	gpu_hash_insert((bucket_t *)device_hash_table, 
        (ielem_t **)blk_input_d, (int *)blk_elem_num_d, 
        INSERT_BLOCK, PRELOAD_CNT, fd, 
        0, operation_time, memcpy_time, pwrite_time, persist_time);
        
	// verify with search
    CUDA_SAFE_CALL(cudaMemcpy(device_search_in, host_search_in, PRELOAD_CNT * sizeof(selem_t), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemset((void *)device_search_out, 1, 2 * PRELOAD_CNT * sizeof(loc_t)));

	printf("Verify %d keys\n", PRELOAD_CNT);
    // ---------------------------
    gpu_hash_search((selem_t *)device_search_in, (loc_t *)device_search_out, 
            (bucket_t *)device_hash_table, PRELOAD_CNT, THREAD_NUM, 128, 0);
    cudaDeviceSynchronize();
    // ---------------------------
    
    CUDA_SAFE_CALL(cudaMemcpy(host_search_out, device_search_out, 
                2 * PRELOAD_CNT * sizeof(loc_t), cudaMemcpyDeviceToHost));

    for (i = 0; i < PRELOAD_CNT; i ++) {
    	loc_t loc = cpu_map[selem_t(((ielem_t *)host_in)[i].sig, ((ielem_t *)host_in)[i].hash)];
        if(((loc_t *)host_search_out)[i<<1] != loc
                && ((loc_t *)host_search_out)[(i<<1)+1] != loc) {
            printf("not found insertion %d : out %lx and %lx, should be : %lx\n", i,
                    ((loc_t *)host_search_out)[i<<1], ((loc_t *)host_search_out)[(i<<1)+1],
                    loc);
        }        	
    }
	
	// Free memory for preload
	CUDA_SAFE_CALL(cudaFree(device_in));
	CUDA_SAFE_CALL(cudaFree(blk_input_d));
	CUDA_SAFE_CALL(cudaFree(blk_elem_num_d));
	CUDA_SAFE_CALL(cudaFree(device_search_in));
	CUDA_SAFE_CALL(cudaFree(device_search_out));
	CUDA_SAFE_CALL(cudaFreeHost(host_in));
	CUDA_SAFE_CALL(cudaFreeHost(host_search_in));
	CUDA_SAFE_CALL(cudaFreeHost(host_search_out));
	CUDA_SAFE_CALL(cudaFreeHost(host_search_verify));
	
	// Allocate for actual insert/searches
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

    int lower_bond;
	ins_time = 0, search_time = 0, del_time = 0;
	persist_time = 0, operation_time = 0, memcpy_time = 0, pwrite_time = 0; 
    int num_ops = 100;
    int num_get = 95;
    int num_set = num_ops - num_get;
    for (int has = 0; has < num_ops; has++) {
    	int selection = rand() % (num_get + num_set);    
		if(selection < num_set) {
			--num_set;
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
    			//cpu_map[selem_t(i+1, i+1)] = ((ielem_t *)host_in)[i].loc;
		        //printf("%d\n", ((int *)host_search_verify)[i]);
		    }
		    //for debugging
		    for (i = 0; i < SELEM_NUM; i += 1) {
		        //printf("%d %d %d\n", ((int *)host_in)[i*3], (i/BLOCK_ELEM_NUM) * BLOCK_ELEM_NUM, 
		        //(i/BLOCK_ELEM_NUM) * BLOCK_ELEM_NUM + BLOCK_ELEM_NUM);
		        assert(((ielem_t *)host_in)[i].hash < (i/BLOCK_ELEM_NUM) * HASH_BLOCK_ELEM_NUM + HASH_BLOCK_ELEM_NUM);
		        assert(((ielem_t *)host_in)[i].hash >= (i/BLOCK_ELEM_NUM) * HASH_BLOCK_ELEM_NUM);
		    }

			auto start_time = TIME_NOW;
		    CUDA_SAFE_CALL(cudaMemcpy(device_in, host_in, SELEM_NUM * sizeof(ielem_t), cudaMemcpyHostToDevice));
		    cudaDeviceSynchronize();
		    gpu_hash_insert((bucket_t *)device_hash_table, 
	            (ielem_t **)blk_input_d,
	            (int *)blk_elem_num_d, INSERT_BLOCK, SELEM_NUM, fd, 
				0, operation_time, memcpy_time, pwrite_time, persist_time);
		    CUDA_SAFE_CALL(cudaDeviceSynchronize());
		    ins_time += time_diff(TIME_NOW, start_time)/ 1000.0f;
		    OUTPUT_STATS

			printf("Batch %d. INSERT: insert %f ms, search %f ms\n", has, ins_time, search_time);
		    /* +++++++++++++++++++++++++++++++++++ SEARCH +++++++++++++++++++++++++++++++++ */

		    // verify with search
		    CUDA_SAFE_CALL(cudaMemcpy(device_search_in, host_search_in, 
		                SELEM_NUM * sizeof(selem_t), cudaMemcpyHostToDevice));
		    CUDA_SAFE_CALL(cudaMemset((void *)device_search_out, 0, 2 * SELEM_NUM * sizeof(loc_t)));
		    // ---------------------------
		    gpu_hash_search((selem_t *)device_search_in, (loc_t *)device_search_out, 
		            (bucket_t *)device_hash_table, SELEM_NUM, THREAD_NUM, 128, 0);
		    cudaDeviceSynchronize();
		    // ---------------------------
		    CUDA_SAFE_CALL(cudaMemcpy(host_search_out, device_search_out, 
		                2 * SELEM_NUM * sizeof(loc_t), cudaMemcpyDeviceToHost));

		    for (i = 0; i < SELEM_NUM; i ++) {
		        if(((loc_t *)host_search_out)[i<<1] != ((loc_t *)host_search_verify)[i] 
		                && ((loc_t *)host_search_out)[(i<<1)+1] != ((loc_t *)host_search_verify)[i]) {
		            printf("not found insertion %d : out %lx and %lx, should be : %lx\n", i,
		                    ((loc_t *)host_search_out)[i<<1], ((loc_t *)host_search_out)[(i<<1)+1],
		                    ((loc_t *)host_search_verify)[i]);
		            assert(false);
		        }
		    }
        }
        else {
        	--num_get;
        	/* +++++++++++++++++++++++++++++++++++ SEARCH +++++++++++++++++++++++++++++++++ */
		    for (i = 0; i < SELEM_NUM; i += 1) {
		        lower_bond = (i / BLOCK_ELEM_NUM) * HASH_BLOCK_ELEM_NUM;
		    	uint32_t get_key = (uint32_t)mehcached_zipf_next(&zipf_state) + 1;
		    	assert(get_key < PRELOAD_CNT);
		        // sig
		        ((selem_t *)host_search_in)[i].sig = get_key;
		        // hash
		        ((selem_t *)host_search_in)[i].hash = get_key;
		    }
		    
		    auto search_start = TIME_NOW;
		    CUDA_SAFE_CALL(cudaMemcpy(device_search_in, host_search_in, 
		                SELEM_NUM * sizeof(selem_t), cudaMemcpyHostToDevice));
		    CUDA_SAFE_CALL(cudaMemset((void *)device_search_out, 0, 2 * SELEM_NUM * sizeof(loc_t)));
		    // ---------------------------
		    gpu_hash_search((selem_t *)device_search_in, (loc_t *)device_search_out, 
		            (bucket_t *)device_hash_table, SELEM_NUM, THREAD_NUM, 128, 0);
		    cudaDeviceSynchronize();
		    // ---------------------------		    
		    CUDA_SAFE_CALL(cudaMemcpy(host_search_out, device_search_out, 
		                2 * SELEM_NUM * sizeof(loc_t), cudaMemcpyDeviceToHost));
		    search_time += (double)time_diff(TIME_NOW, search_start) / 1000.0;

		    for (i = 0; i < SELEM_NUM; i ++) {
				loc_t loc = cpu_map[selem_t(((selem_t *)host_search_in)[i].sig, ((selem_t *)host_search_in)[i].hash)];
				if(((loc_t *)host_search_out)[i<<1] != loc
				        && ((loc_t *)host_search_out)[(i<<1)+1] != loc) {
		            printf("not found insertion %d, key %d : out %lx and %lx, should be : %lx\n", i, 
		            	((selem_t *)host_search_in)[i].sig, ((loc_t *)host_search_out)[i<<1], 
		            	((loc_t *)host_search_out)[(i<<1)+1], loc);
		            assert(false);
		        }
		    }
			printf("Batch %d. SEARCH: insert %f ms, search %f ms\n", has, ins_time, search_time);
        }
    }
    
    printf("\nOperation execution time: %f ms\n", operation_time/1000000.0);  
    printf("MemcpyTime\t%f\tms\nPersistTime\t%f\nPwriteTime\t%f\n", memcpy_time/1000000.0, persist_time/1000000.0, pwrite_time/1000000.0);

    printf("\n\n");
    printf("Insert: %f ms, search: %f ms\n", ins_time, search_time);
    printf("Runtime\t%f\tms\n", ins_time + search_time);

    return 0;
}
