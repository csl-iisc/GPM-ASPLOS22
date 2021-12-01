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

#define GPUDB

extern "C" 
{
#include "change-ddio.h"
}
#include "support/cuda-setup.h"
#include "kernel_mm.h"
#include "support/common.h"
#include "support/timer.h"
#include "support/verify.h"
#include "bandwidth_analysis.cuh"
#include <unistd.h>
#include <thread>
#include <assert.h>
#include <libpmem.h>
#include <chrono>
#include "libgpm.cuh"

double operation_time = 0, memcpy_time = 0, persist_time = 0; 
long long nvm_writes = 0; 

#define TIME_NOW std::chrono::high_resolution_clock::now()  

// Params ---------------------------------------------------------------------
struct Params {

    int         device;
    int         n_gpu_threads;
    int         n_gpu_blocks;
    int         n_threads;
    int         n_warmup;
    int         n_reps;
    const char *file_name;
    const char *comparison_file;
    int         switching_limit;

    Params(int argc, char **argv) {
        device          = 0;
        n_gpu_threads    = 256;
        n_gpu_blocks   = 24;
        n_threads       = 2;
        n_warmup        = 1;
        n_reps          = 1;
        file_name       = "input/USA-road-d.USA.gr.parboil";
        comparison_file = "output/NYR_bfs_BFS.out";
        switching_limit = 0;
        int opt;
        while((opt = getopt(argc, argv, "hd:i:g:t:w:r:f:c:l:")) >= 0) {
            switch(opt) {
                case 'h':
                    usage();
                    exit(0);
                    break;
                case 'd': device          = atoi(optarg); break;
                case 'i': n_gpu_threads    = atoi(optarg); break;
                case 'g': n_gpu_blocks   = atoi(optarg); break;
                case 't': n_threads       = atoi(optarg); break;
                case 'w': n_warmup        = atoi(optarg); break;
                case 'r': n_reps          = atoi(optarg); break;
                case 'f': file_name       = optarg; break;
                case 'c': comparison_file = optarg; break;
                case 'l': switching_limit = atoi(optarg); break;
                default:
                          fprintf(stderr, "\nUnrecognized option!\n");
                          usage();
                          exit(0);
            }
        }
        assert(n_gpu_threads > 0 && "Invalid # of device threads!");
        assert(n_gpu_blocks > 0 && "Invalid # of device blocks!");
        assert(n_threads > 0 && "Invalid # of host threads!");
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./bfs [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -d <D>    CUDA device ID (default=0)"
                "\n    -i <I>    # of device threads per block (default=256)"
                "\n    -g <G>    # of device blocks (default=8)"
                "\n    -t <T>    # of host threads (default=2)"
                "\n    -w <W>    # of untimed warmup iterations (default=1)"
                "\n    -r <R>    # of timed repetition iterations (default=1)"
                "\n"
                "\nBenchmark-specific options:"
                "\n    -f <F>    name of input file with control points (default=input/NYR_input.dat)"
                "\n    -c <C>    comparison file (default=output/NYR_bfs_BFS.out)"
                "\n    -l <L>    switching limit (default=128)"
                "\n");
    }
};

// Input Data -----------------------------------------------------------------
void read_input_size(int &n_nodes, int &n_edges, const Params &p) {
    FILE *fp = fopen(p.file_name, "r");
    CHECK(fscanf(fp, "%d", &n_nodes))
        CHECK(fscanf(fp, "%d", &n_edges))
        if(fp)
            fclose(fp);
}


void read_input(int &source, Node *&h_nodes, Edge *&h_edges, const Params &p) {

    int   start, edgeno;
    int   n_nodes, n_edges;
    int   id, cost;
    FILE *fp = fopen(p.file_name, "r");

    CHECK(fscanf(fp, "%d", &n_nodes))
        CHECK(fscanf(fp, "%d", &n_edges))
        CHECK(fscanf(fp, "%d", &source))
        printf("Number of nodes = %d\n", n_nodes);
    printf("Number of edges = %d\n", n_edges);

    // initalize the memory: Nodes
    for(int i = 0; i < n_nodes; i++) {
        CHECK(fscanf(fp, "%d %d", &start, &edgeno))
            h_nodes[i].x = start;
        h_nodes[i].y = edgeno;
    }
#if PRINT_ALL
    for(int i = 0; i < n_nodes; i++) {
        CHECK(printf("%d, %d\n", h_nodes[i].x, h_nodes[i].y))
    }
#endif

    // initalize the memory: Edges
    for(int i = 0; i < n_edges; i++) {
        CHECK(fscanf(fp, "%d", &id))
            CHECK(fscanf(fp, "%d", &cost))
            h_edges[i].x = id;
        h_edges[i].y = -cost;
    }
    if(fp)
        fclose(fp);
}

const bool recovery = false;
// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {

    const Params p(argc, argv);
    CUDASetup    setcuda(p.device);
    //Timer        timer;
    cudaError_t  cudaStatus;

    int n_nodes, n_edges;
    read_input_size(n_nodes, n_edges, p);

    size_t len_q2   = n_nodes * sizeof(int);
    size_t len_q1   = n_nodes * sizeof(int);
    size_t len_cost = n_nodes * sizeof(std::atomic_int);
    size_t len_color = n_nodes * sizeof(std::atomic_int);
    size_t len_iter = sizeof(int);
    size_t len_size = sizeof(int);
   
    //pm_nodes and pm_edges contain the input data and both coarse and fine-grained read from it 
    //pm_nodes and pm_edges are copied to d_nodes and d_edges for all cases 
    //d_cost etc are written back to pm_cost only for coarse-grained
    // Allocate
    //timer.start("Allocation");
    size_t mapped_len; 
    int is_pmemp; 
    Node * h_nodes = (Node *)malloc(sizeof(Node) * n_nodes);
    Node * d_nodes;
    cudaStatus = cudaMalloc((void**)&d_nodes, sizeof(Node) * n_nodes);
    //Persistent input 
    Edge * h_edges = (Edge *)malloc(sizeof(Edge) * n_edges);
    Edge * d_edges;
    //Persistent input 
    cudaStatus = cudaMalloc((void**)&d_edges, sizeof(Edge) * n_edges);
    std::atomic_int *h_color = (std::atomic_int *)malloc(sizeof(std::atomic_int) * n_nodes);
    int * d_color;
    cudaStatus = cudaMalloc((void**)&d_color, sizeof(int) * n_nodes);
    std::atomic_int *h_cost  = (std::atomic_int *)malloc(sizeof(std::atomic_int) * n_nodes);
    int * d_cost;
    cudaStatus = cudaMalloc((void**)&d_cost, sizeof(int) * n_nodes);
    int *            h_q1    = (int *)malloc(n_nodes * sizeof(int));
    int * d_q1;
    cudaStatus = cudaMalloc((void**)&d_q1, sizeof(int) * n_nodes);
    int *            h_q2    = (int *)malloc(n_nodes * sizeof(int));
    int * d_q2;
    cudaStatus = cudaMalloc((void**)&d_q2, sizeof(int) * n_nodes);
    std::atomic_int  h_head[1];
    int * d_head;
    cudaStatus = cudaMalloc((void**)&d_head, sizeof(int));
    std::atomic_int  h_tail[1];
    int * d_tail;
    cudaStatus = cudaMalloc((void**)&d_tail, sizeof(int));
    std::atomic_int  h_threads_end[1];
    int * d_threads_end;
    cudaStatus = cudaMalloc((void**)&d_threads_end, sizeof(int));
    std::atomic_int  h_threads_run[1];
    int * d_threads_run;
    cudaStatus = cudaMalloc((void**)&d_threads_run, sizeof(int));
    int              h_num_t[1];
    int * d_num_t;
    cudaStatus = cudaMalloc((void**)&d_num_t, sizeof(int));
    int              h_overflow[1];
    int * d_overflow;
    cudaStatus = cudaMalloc((void**)&d_overflow, sizeof(int));
    std::atomic_int  h_iter[1];
    int * d_iter;
    cudaStatus = cudaMalloc((void**)&d_iter, sizeof(int));
    ALLOC_ERR(h_nodes, h_edges, h_color, h_cost, h_q1, h_q2);
    CUDA_ERR();
    cudaDeviceSynchronize();
    //timer.stop("Allocation");
    //Persistent output 
    int *pm_cost = (int*) pmem_map_file("/pmem/pm_cost", len_cost, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmemp);
    int *pm_q1 = (int*) pmem_map_file("/pmem/pm_q1", len_q1, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmemp);
    int *pm_q2 = (int*) pmem_map_file("/pmem/pm_q2", len_q2, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmemp);
    int *pm_num_t = (int*) pmem_map_file("/pmem/pm_num_t", len_size, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmemp);
    int *pm_iter = (int*) pmem_map_file("/pmem/pm_iter", len_iter, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmemp);
    int *pm_color = (int*) pmem_map_file("/pmem/pm_color", len_color, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmemp);
        // Initialize
    //timer.start("Initialization");
    const int max_gpu_threads = setcuda.max_gpu_threads();
    int source;
    read_input(source, h_nodes, h_edges, p);
    for(int i = 0; i < n_nodes; i++) {
        //Cost was already initialized
        h_cost[i].store(INF);
    }
    h_cost[source].store(0);
    for(int i = 0; i < n_nodes; i++) {
        h_color[i].store(WHITE);
    }
    h_tail[0].store(0);
    h_head[0].store(0);
    h_threads_end[0].store(0);
    h_threads_run[0].store(0);
    h_q1[0] = source;
    h_iter[0].store(0);
    h_overflow[0] = 0;
    cudaDeviceSynchronize();


    char *ptr = getenv("PMEM_THREADS");
	size_t threads;
	if(ptr != NULL)
		threads = atoi(ptr);
	else
		threads = 1;
    // Copy to device
    //timer.start("Copy To Device");

    auto start = TIME_NOW; 
    cudaStatus =
        cudaMemcpy(d_nodes, h_nodes, sizeof(Node) * n_nodes, cudaMemcpyHostToDevice);
    cudaStatus =
        cudaMemcpy(d_edges, h_edges, sizeof(Edge) * n_edges, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    memcpy_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
    CUDA_ERR();
    //timer.stop("Copy To Device");
   for(int rep = 0; rep < p.n_reps + p.n_warmup; rep++) {

        // Reset
        for(int i = 0; i < n_nodes; i++) {
            h_cost[i].store(INF);
        }
        h_cost[source].store(0);
        for(int i = 0; i < n_nodes; i++) {
            h_color[i].store(WHITE);
        }
        h_tail[0].store(0);
        h_head[0].store(0);
        h_threads_end[0].store(0);
        h_threads_run[0].store(0);
        h_q1[0] = source;
        h_iter[0].store(0);
        h_overflow[0] = 0;

        /*if(rep >= p.n_warmup)
          timer.start("Kernel");*/

        // Run first iteration in master CPU thread
        h_num_t[0] = 1;
        int pid;
        int index_i, index_o;
        for(index_i = 0; index_i < h_num_t[0]; index_i++) {
            pid = h_q1[index_i];
            h_color[pid].store(BLACK);
            for(int i = h_nodes[pid].x; i < (h_nodes[pid].y + h_nodes[pid].x); i++) {
                int id = h_edges[i].x;
                h_color[id].store(BLACK);
                index_o       = h_tail[0].fetch_add(1);
                h_q2[index_o] = id;
            }
        }
        h_num_t[0] = h_tail[0].load();
        h_tail[0].store(0);
        h_threads_run[0].fetch_add(1);
        h_iter[0].fetch_add(1);

        // Pointers to input and output queues
        int * d_qin  = d_q2;
        int * d_qout = d_q1;

        const int CPU_EXEC = 0;//(p.n_threads > 0) ? 1 : 0;

        //Copy Reset values 
        cudaMemcpy(d_cost, h_cost, n_nodes * sizeof(int), cudaMemcpyHostToDevice); 
        cudaMemcpy(d_color, h_color, n_nodes * sizeof(int), cudaMemcpyHostToDevice); 
        cudaMemcpy(d_q1, h_q1, n_nodes * sizeof(int), cudaMemcpyHostToDevice); 
        cudaMemcpy(d_q2, h_q2, n_nodes * sizeof(int), cudaMemcpyHostToDevice); 
        cudaMemcpy(d_overflow, h_overflow, sizeof(int), cudaMemcpyHostToDevice); 
        cudaMemcpy(d_iter, h_iter, sizeof(int), cudaMemcpyHostToDevice); 
        cudaMemcpy(d_threads_run, h_threads_run, sizeof(int), cudaMemcpyHostToDevice); 
        cudaMemcpy(d_threads_end, h_threads_end, sizeof(int), cudaMemcpyHostToDevice); 
        //START_BW_MONITOR2("bw_mm_bfs.csv")
        //Run BFS
        while(*h_num_t != 0) {
            if(h_iter[0] % 2 == 0) {
                d_qin  = d_q1;
                d_qout = d_q2;
            } else {
                d_qin  = d_q2;
                d_qout = d_q1;
            }
            cudaMemcpy(d_num_t, h_num_t, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_tail, h_tail, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_head, h_head, sizeof(int), cudaMemcpyHostToDevice);
            start = std::chrono::high_resolution_clock::now(); 
            cudaStatus = call_BFS_gpu(p.n_gpu_blocks, p.n_gpu_threads, d_nodes, d_edges,d_cost,
                    d_color, d_qin, d_qout, d_num_t,
                    d_head, d_tail, d_threads_end, d_threads_run,
                    d_overflow, d_iter, p.switching_limit, CPU_EXEC, sizeof(int) * (W_QUEUE_SIZE + 3));
		    cudaDeviceSynchronize();
		    operation_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
		    
           //CUDA_ERR();
            //cudaMemcpy(h_q1, d_q1, sizeof(int) * n_nodes, cudaMemcpyDeviceToHost); 
            //cudaMemcpy(h_q2, d_q2, sizeof(int) * n_nodes, cudaMemcpyDeviceToHost); 
            cudaMemcpy(h_num_t, d_num_t, sizeof(int), cudaMemcpyDeviceToHost); 
            cudaMemcpy(h_iter, d_iter, sizeof(int), cudaMemcpyDeviceToHost); 
            cudaMemcpy(h_tail, d_tail, sizeof(int), cudaMemcpyDeviceToHost); 
            cudaMemcpy(h_head, d_head, sizeof(int), cudaMemcpyDeviceToHost); 
            
	    	int *pm_qout;
	    	if(d_qout == d_q1)
	        	pm_qout = pm_q1;
	        else
	        	pm_qout = pm_q2;
	        
	    	size_t sz = h_tail[0];
	    	
		    start = std::chrono::high_resolution_clock::now();
		    //cudaMemcpy(pm_cost, d_cost, sizeof(int) * n_nodes, cudaMemcpyDeviceToHost);
	    	cudaMemcpy(pm_qout, d_qout, sizeof(int) * sz, cudaMemcpyDeviceToHost);
		    for(int i = 0; i < sz; ++i) {
		    	cudaMemcpy(&pm_cost[pm_qout[i]], &d_cost[pm_qout[i]], sizeof(int), cudaMemcpyDeviceToHost); 
		    	cudaMemcpy(&pm_color[pm_qout[i]], &d_color[pm_qout[i]], sizeof(int), cudaMemcpyDeviceToHost); 
		    }
		    cudaMemcpy(pm_num_t, d_num_t, sizeof(int), cudaMemcpyDeviceToHost); 
		    cudaMemcpy(pm_iter, d_iter, sizeof(int), cudaMemcpyDeviceToHost); 
		    memcpy_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
            
		    start = TIME_NOW; 
		    if (is_pmemp) {
		    	if(d_qout == d_q1) {
		        	pmem_mt_persist(pm_q1, sizeof(int) * sz);
                    nvm_writes += sizeof(int) * sz; 
		        }
		        else { 
		        	pmem_mt_persist(pm_q2, sizeof(int) * sz);
                    nvm_writes += sizeof(int) * sz; 
		        }
		        
    			#pragma omp parallel for num_threads(threads)
			    for(int i = 0; i < sz; ++i) {
			    	pmem_persist(&pm_cost[pm_qout[i]], sizeof(int));
			    	pmem_persist(&pm_color[pm_qout[i]], sizeof(int));
                    nvm_writes += 2 * sizeof(int); 
			    }
		        pmem_persist(pm_num_t, sizeof(int)); 
                nvm_writes += sizeof(int); 
		        pmem_persist(pm_iter, sizeof(int)); 
                nvm_writes += sizeof(int); 
		    }
		    else {
		        pmem_msync(pm_cost, sizeof(int) * n_nodes); 
		        pmem_msync(pm_q1, sizeof(int) * n_nodes); 
		        pmem_msync(pm_q2, sizeof(int) * n_nodes); 
		        pmem_msync(pm_num_t, sizeof(int)); 
		        pmem_msync(pm_iter, sizeof(int)); 
		    }
		    persist_time += (TIME_NOW - start).count(); 
		    
            int x = h_iter[0];
            if(x % 100 == 0)
                printf("%d\n", x);
            
            h_num_t[0] = h_tail[0].load(); // Number of elements in output queue
            h_tail[0].store(0);
            h_head[0].store(0);
        }
        //STOP_BW_MONITOR
        cudaMemcpy(h_iter, d_iter, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_cost, d_cost, sizeof(int) * n_nodes, cudaMemcpyDeviceToHost); 
        //BFS done running
      //CUDA_ERR();
    }
printf("\nOperation execution time: %f ms\n", operation_time/1000000.0);
printf("memcpy time: %f \t persist time: %f \t\n", memcpy_time/1000000.0f, persist_time/1000000.0f); 
printf("\nruntime: %f ms\n", (operation_time + memcpy_time + persist_time)/1000000.0);
printf("Tot writes: %lli\n", nvm_writes); 

// Verify answer
//verify(h_cost, n_nodes, p.comparison_file);

// Free memory
free(h_nodes);
free(h_edges);
free(h_color);
free(h_cost);
free(h_q1);
free(h_q2);
cudaStatus = cudaFree(d_color);
cudaStatus = cudaFree(d_head);
cudaStatus = cudaFree(d_tail);
cudaStatus = cudaFree(d_threads_end);
cudaStatus = cudaFree(d_threads_run);
cudaStatus = cudaFree(d_overflow);
cudaStatus = cudaFree(d_iter);
//CUDA_ERR();
cudaDeviceSynchronize();

printf("Test Passed\n");
return 0;
}
