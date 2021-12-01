/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#define _CUDA_COMPILER_
#define GPUDB
#include "support/common.h"
#include <stdio.h>
#include "libgpm.cuh"
#include <cooperative_groups.h>
using namespace cooperative_groups;

#ifdef OUTPUT_NVM_DETAILS
	__device__ uint64_t nvm_write;
	__device__ uint64_t nvm_read;
#endif

__global__ void BFS_gpu(Node *graph_nodes_av, Edge *graph_edges_av, int *cost,
        int *color, int *q1, int *q2, int n_t,
        int *head, int *tail, int *threads_end, int *threads_run,
        int *overflow, int *iter, int LIMIT, const int CPU, int h_iter, int cleanup = false) {
    extern __shared__ int l_mem[];
    int* tail_bin = l_mem;
    int* l_q2 = (int*)&tail_bin[1];
    int* shift = (int*)&l_q2[W_QUEUE_SIZE];
    int* base = (int*)&shift[1];

    const int tid     = threadIdx.x;
    const int gtid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int MAXWG   = gridDim.x;
    const int WG_SIZE = blockDim.x;

    int iter_local = h_iter;
    //PMEM_READ_OP(, sizeof(int)) //For num_t which is also persistent 
    int n_t_local = n_t;

	grid_group g = this_grid();
	
	if(cleanup) {
		// Fetch frontier elements from the queue
		if(tid == 0)
		    *base = atomicAdd(&head[0], WG_SIZE);
		__syncthreads();

		int my_base = *base;
		while(my_base < n_t_local) {
		    if(my_base + tid < n_t_local) {
		        // Visit a node from the current frontier
		        PMEM_READ_OP(, sizeof(int))
		        int pid = q1[my_base + tid];
		        Node cur_node;
		        cur_node.x = graph_nodes_av[pid].x;
		        cur_node.y = graph_nodes_av[pid].y;
		        for(int i = cur_node.x; i < cur_node.y + cur_node.x; i++) {
		            int id = graph_edges_av[i].x; 
		            //int old_color = color[id];
		            PMEM_READ_OP(, sizeof(int))
		            if(color[id] == iter_local) {
		            	PMEM_WRITE_OP(color[id] = INF, sizeof(int))
		            }
		        }
		    }
		    if(tid == 0)
		        *base = atomicAdd(&head[0], WG_SIZE); // Fetch more frontier elements from the queue
		    __syncthreads();
		    my_base = *base;
		}
	}
	
    g.sync();
    
    if(gtid == 0) {
        head[0] = 0;
    }
    g.sync();
    
    if(tid == 0) {
        // Reset queue
        *tail_bin = 0;
    }

    // Fetch frontier elements from the queue
    if(tid == 0)
        *base = atomicAdd(&head[0], WG_SIZE);
    __syncthreads();

    int my_base = *base;
    while(my_base < n_t_local) {
        if(my_base + tid < n_t_local && *overflow == 0) {
            // Visit a node from the current frontier
            PMEM_READ_OP(, sizeof(int))
            int pid = q1[my_base + tid];
            //////////////// Visit node ///////////////////////////
            /*
               atomicExch(&cost[pid], iter_local); // Node visited
               gpm_persist(cost + pid, sizeof(int));
               atomicAdd(d_dpoints, 1);
               atomicAdd(d_opoints, 1);
             */
            //atomicExch(&cost[pid], iter_local); // Node visited
            Node cur_node;
            //PMEM_READ_OP( cur_node.x = graph_nodes_av[pid].x , 4 )
            //PMEM_READ_OP( cur_node.y = graph_nodes_av[pid].y , 4 )
            cur_node.x = graph_nodes_av[pid].x; 
            cur_node.y = graph_nodes_av[pid].y; 
            for(int i = cur_node.x; i < cur_node.y + cur_node.x; i++) {
                //PMEM_READ_OP ( int id        = graph_edges_av[i].x , 4 )
                int id = graph_edges_av[i].x; 
                PMEM_READ_OP(int old_color = atomicMin_system(&color[id], iter_local), sizeof(int) )
                if(old_color > iter_local) {
                	PMEM_WRITE_OP( , sizeof(int) )
                    // Push to the queue
                    int tail_index = atomicAdd(tail_bin, 1);
                    if(tail_index >= W_QUEUE_SIZE) {
                        *overflow = 1;
                    } else
                        l_q2[tail_index] = id;
                }
            }
            gpm_memcpy_nodrain(&cost[pid], &iter_local, sizeof(int), cudaMemcpyDeviceToDevice);
        }
        if(tid == 0)
            *base = atomicAdd(&head[0], WG_SIZE); // Fetch more frontier elements from the queue
        __syncthreads();
        my_base = *base;
    }
    /////////////////////////////////////////////////////////
    // Compute size of the output and allocate space in the global queue
    if(tid == 0) {
        *shift = atomicAdd(&tail[0], *tail_bin);
    }
//    gpm_drain();
    __syncthreads();
    ///////////////////// CONCATENATE INTO GLOBAL MEMORY /////////////////////
    int local_shift = tid;
    while(local_shift < *tail_bin) {
        gpm_memcpy_nodrain(&q2[*shift + local_shift], &l_q2[local_shift], sizeof(int), cudaMemcpyDeviceToDevice);
        //gpm_persist(&q2[*shift + local_shift], sizeof(int));
        /* 
           gpm_persist(q2 + *shift + local_shift, sizeof(int));
           atomicAdd(d_dpoints, 1);
           atomicAdd(d_opoints, 1);
         */
        //gpm_persist(q2 + *shift + local_shift, sizeof(int));
        // Multiple threads are copying elements at the same time, so we shift by multiple elements  for next iteration
        local_shift += WG_SIZE;
/*    }
    g.sync();
    //////////////////////////////////////////////////////////////////////////
    if(gtid == 0) {
        PMEM_READ_WRITE_OP( atomicAdd(&iter[0], 1) , sizeof(int) )*/
    }
    gpm_drain();
}

cudaError_t call_BFS_gpu(int blocks, int threads, Node *graph_nodes_av, Edge *graph_edges_av, int *cost,
    int *color, int *q1, int *q2, int n_t,
    int *head, int *tail, int *threads_end, int *threads_run,
    int *overflow, int *iter, int LIMIT, const int CPU, int l_mem_size, int h_iter, int cleanup = 0){

    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);
    void* params[18];
    int i = 0;
    params[i++] = (void*)&graph_nodes_av;
    params[i++] = (void*)&graph_edges_av;
    params[i++] = (void*)&cost;
    params[i++] = (void*)&color;
    params[i++] = (void*)&q1;
    params[i++] = (void*)&q2;
    params[i++] = (void*)&n_t;
    params[i++] = (void*)&head;
    params[i++] = (void*)&tail;
    params[i++] = (void*)&threads_end;
    params[i++] = (void*)&threads_run;
    params[i++] = (void*)&overflow;
    params[i++] = (void*)&iter;
    params[i++] = (void*)&LIMIT;
    params[i++] = (void*)&CPU;
    params[i++] = (void*)&h_iter;
    params[i++] = (void*)&cleanup;
    
    cudaLaunchCooperativeKernel ( (const void*) BFS_gpu, dimGrid, dimBlock, params, l_mem_size, 0 );
    //BFS_gpu<<<dimGrid, dimBlock, l_mem_size>>>(graph_nodes_av, graph_edges_av, cost,
    //    color, q1, q2, n_t,
    //    head, tail, threads_end, threads_run,
    //    overflow, iter, LIMIT, CPU, h_iter);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    return err;
}


