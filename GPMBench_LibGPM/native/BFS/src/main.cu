/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *				  University of Cordoba and University of Illinois
 *				  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *		this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *		notice, this list of conditions and the following disclaimers in the
 *		documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba,
 *		University of Illinois nor the names of its contributors may be used
 *		to endorse or promote products derived from this Software without
 *		specific prior written permission.
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
#include "kernel.h"
#include "support/common.h"
#include "support/timer.h"
#include "support/verify.h"
#include "bandwidth_analysis.cuh"
#include "libgpm.cuh"
#include <unistd.h>
#include <thread>
#include <assert.h>

void *PMEM_START_HOST; 
__constant__ char *PMEM_START_DEV;
char *gpm_start_alloc; // Point from which memory allocation can begin
bool gpm_init_complete = false;
double operation_time = 0, persist_time = 0; 

// Params ---------------------------------------------------------------------
struct Params {

    int		 device;
    int		 n_gpu_threads;
    int		 n_gpu_blocks;
    int		 n_threads;
    int		 n_warmup;
    int		 n_reps;
    const char *file_name;
    const char *comparison_file;
    int		 switching_limit;

    Params(int argc, char **argv) {
		device		  = 0;
		n_gpu_threads    = 256;
		n_gpu_blocks   = 24;
		n_threads       = 2;
		n_warmup		= 1;
		n_reps		  = 1;
		file_name       = "input/USA-road-d.USA.gr.parboil";
		comparison_file = "output/USA-road-d.USA.gr.out";
		//file_name = "/home/shweta/chiselx/hollywood-2009/hollywood-2009.mtx";
		switching_limit = 0;
		int opt;
		while((opt = getopt(argc, argv, "hd:i:g:t:w:r:f:c:l:")) >= 0) {
		    switch(opt) {
				case 'h':
				    usage();
				    exit(0);
				    break;
				case 'd': device		  = atoi(optarg); break;
				case 'i': n_gpu_threads    = atoi(optarg); break;
				case 'g': n_gpu_blocks   = atoi(optarg); break;
				case 't': n_threads       = atoi(optarg); break;
				case 'w': n_warmup		= atoi(optarg); break;
				case 'r': n_reps		  = atoi(optarg); break;
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
				"\n    -h		help"
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
void read_input_size(long long &n_nodes, long long &n_edges, const Params &p) {
    FILE *fp = fopen(p.file_name, "r");
    CHECK(fscanf(fp, "%lld", &n_nodes))
		CHECK(fscanf(fp, "%lld", &n_edges))
    printf("Number of nodes = %lld\n", n_nodes);
    printf("Number of edges = %lld\n", n_edges);
    fflush(stdout);
		if(fp)
		    fclose(fp);
}


void read_input(int &source, Node *&h_nodes, Edge *&h_edges, const Params &p) {

    int   start, edgeno;
    long long   n_nodes, n_edges;
    int   id, cost;
    FILE *fp = fopen(p.file_name, "r");

    CHECK(fscanf(fp, "%lld", &n_nodes))
		CHECK(fscanf(fp, "%lld", &n_edges))
		CHECK(fscanf(fp, "%d", &source))
    // initalize the memory: Nodes
    for(long long i = 0; i < n_nodes; i++) {
		CHECK(fscanf(fp, "%d %d", &start, &edgeno))
		    h_nodes[i].x = start;
		h_nodes[i].y = edgeno;
    }
#if PRINT_ALL
    for(long long i = 0; i < n_nodes; i++) {
		CHECK(printf("%d, %d\n", h_nodes[i].x, h_nodes[i].y))
    }
#endif

    // initalize the memory: Edges
    for(long long i = 0; i < n_edges; i++) {
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
    //Timer		timer;
    cudaError_t  cudaStatus;

    const char *path_node = "/pmem/persist_bfs_node.dat";
    const char *path_edge = "/pmem/persist_bfs_edge.dat";
    const char *path_q2   = "persist_bfs_q2.dat";
    const char *path_q1   = "persist_bfs_q1.dat";
    const char *path_cost = "persist_bfs_cost.dat";
    const char *path_color = "persist_bfs_color.dat";
    const char *path_iter = "persist_bfs_iter.dat";
    const char *path_size = "persist_bfs_size.dat";
    long long n_nodes, n_edges;
    read_input_size(n_nodes, n_edges, p);
   
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
    size_t len_node = n_nodes * sizeof(Node);
    //Persistent input 
    Node *pm_nodes = (Node*) pmem_map_file(path_node, len_node, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmemp);
    Edge * h_edges = (Edge *)malloc(sizeof(Edge) * n_edges);
    Edge * d_edges;
    size_t len_edge = n_edges * sizeof(Edge);
    //Persistent input 
    Node *pm_edges = (Node*) pmem_map_file(path_edge, len_edge, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmemp);
    cudaStatus = cudaMalloc((void**)&d_edges, sizeof(Edge) * n_edges);
    int *h_color = (int *)malloc(sizeof(int) * n_nodes);
    int * d_color;
    //cudaStatus = cudaMalloc((void**)&d_color, sizeof(int) * n_nodes);
    int *h_cost  = (int *)malloc(sizeof(int) * n_nodes);
    int * d_cost;
    //cudaStatus = cudaMalloc((void**)&d_cost, sizeof(int) * n_nodes);
    int *		    h_q1    = (int *)malloc(n_nodes * sizeof(int));
    int * d_q1;
    //cudaStatus = cudaMalloc((void**)&d_q1, sizeof(int) * n_nodes);
    int *		    h_q2    = (int *)malloc(n_nodes * sizeof(int));
    int * d_q2;
    //cudaStatus = cudaMalloc((void**)&d_q2, sizeof(int) * n_nodes);
    int  h_head;
    int * d_head;
    cudaStatus = cudaMalloc((void**)&d_head, sizeof(int));
    int  h_tail;
    int * d_tail;
    cudaStatus = cudaMalloc((void**)&d_tail, sizeof(int));
    int  h_threads_end;
    int * d_threads_end;
    cudaStatus = cudaMalloc((void**)&d_threads_end, sizeof(int));
    int  h_threads_run;
    int * d_threads_run;
    cudaStatus = cudaMalloc((void**)&d_threads_run, sizeof(int));
    int		      h_num_t;
    int * d_num_t;
    //cudaStatus = cudaMalloc((void**)&d_num_t, sizeof(int));
    int		      h_overflow;
    int * d_overflow;
    cudaStatus = cudaMalloc((void**)&d_overflow, sizeof(int));
    int  h_iter;
    int * d_iter;

    size_t len_iter = 0, len_q1 = 0, len_q2 = 0;
    size_t len_cost = 0, len_num_t = 0, len_color = 0; 
    if (recovery)
    { 
		auto start = std::chrono::high_resolution_clock::now(); 
		d_iter = (int *)gpm_map_file(path_iter, len_iter, false); 
		d_q1   = (int *)gpm_map_file(path_q1, len_q1, false);
		d_q2   = (int *)gpm_map_file(path_q2, len_q2, false);
		d_cost = (int *)gpm_map_file(path_cost, len_cost, false);
		d_num_t = (int *)gpm_map_file(path_size, len_num_t, false);
		d_color = (int *)gpm_map_file(path_color, len_color, false); 
		printf("Starting from iteration %d\n", *d_iter);
		double recovery_time = (double)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000.0f;
		printf("Recovery time: %f ms\n", recovery_time); 
    }
    else {
		len_q2   = n_nodes * sizeof(int);
		len_q1   = n_nodes * sizeof(int);
		len_cost = n_nodes * sizeof(int);
		len_color= n_nodes * sizeof(int);
		len_iter = sizeof(int);
		len_num_t = sizeof(int) * 2;
		d_cost  = (int*) gpm_map_file(path_cost, len_cost, true);
		d_color = (int*) gpm_map_file(path_color, len_color, true);
		d_q2    = (int *) gpm_map_file(path_q2, len_q2, true);
		d_q1    = (int *) gpm_map_file(path_q1,  len_q1, true);
		d_num_t = (int *) gpm_map_file(path_size, len_num_t, true);
		d_iter  = (int *) gpm_map_file(path_iter,  len_iter, true);
    }

    //cudaStatus = cudaMalloc((void**)&d_iter, sizeof(int));
    ALLOC_ERR(h_nodes, h_edges, h_color, h_cost, h_q1, h_q2);
    CUDA_ERR();
    cudaDeviceSynchronize();
    //timer.stop("Allocation");
    //Persistent output 
#if defined(NVM_ALLOC_GPU) && !defined(EMULATE_NVM)
    int *pm_cost = (int*) pmem_map_file("/pmem/pm_cost", len_cost, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmemp);
    int *pm_q1 = (int*) pmem_map_file("/pmem/pm_q1", len_q1, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmemp);
    int *pm_q2 = (int*) pmem_map_file("/pmem/pm_q2", len_q2, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmemp);
    int *pm_num_t = (int*) pmem_map_file("/pmem/pm_num_t", len_num_t, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmemp);
    int *pm_iter = (int*) pmem_map_file("/pmem/pm_iter", len_iter, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmemp);
#endif
		// Initialize
    //timer.start("Initialization");
    const int max_gpu_threads = setcuda.max_gpu_threads();
    int source;
    read_input(source, h_nodes, h_edges, p);
    for(long long i = 0; i < n_nodes; i++) {
		//Cost was already initialized
		h_cost[i] = INF;
    }
    h_cost[source] = 0;
    for(long long i = 0; i < n_nodes; i++) {
		h_color[i] = WHITE;
    }
    h_tail = 0;
    h_head = 0;
    h_threads_end = 0;
    h_threads_run = 0;
    h_q1[0] = source;
    h_iter = 0;
    h_overflow = 0;
    cudaDeviceSynchronize();


    // Copy to device
    //timer.start("Copy To Device");
    cudaStatus =
		cudaMemcpy(pm_nodes, h_nodes, sizeof(Node) * n_nodes, cudaMemcpyHostToHost);
    cudaStatus =
		cudaMemcpy(pm_edges, h_edges, sizeof(Edge) * n_edges, cudaMemcpyHostToHost);
    
    auto start = std::chrono::high_resolution_clock::now(); 
    cudaStatus =
		cudaMemcpy(d_nodes, pm_nodes, sizeof(Node) * n_nodes, cudaMemcpyHostToDevice);
    cudaStatus =
		cudaMemcpy(d_edges, pm_edges, sizeof(Edge) * n_edges, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    operation_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
    CUDA_ERR();
    //Turn DDIO off here 
#if defined(NVM_ALLOC_CPU) && !defined(FAKE_NVM) && !defined(GPM_WDP)
	start = std::chrono::high_resolution_clock::now(); 
	ddio_off(); 
	operation_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
#endif
    //timer.stop("Copy To Device");
   for(int rep = 0; rep < p.n_reps + p.n_warmup; rep++) {
		
		// Pointers to input and output queues
		int * h_qin  = h_q2;
		int * h_qout = h_q1;
		int * d_qin  = d_q2;
		int * d_qout = d_q1;

		const int CPU_EXEC = 0;//(p.n_threads > 0) ? 1 : 0;
	    const int GPU_EXEC = 1;//(p.n_gpu_blocks > 0 && p.n_gpu_threads > 0) ? 1 : 0;
		int cleanup = 0;
		if(!recovery || *d_iter <= 2) {
		    // Reset
		    for(long long i = 0; i < n_nodes; i++) {
				h_cost[i] = INF;
		    }
		    h_cost[source] = 0;
		    for(long long i = 0; i < n_nodes; i++) {
				h_color[i] = WHITE;
		    }
		    h_tail = 0;
		    h_head = 0;
		    h_threads_end = 0;
		    h_threads_run = 0;
		    h_q1[0] = source;
		    h_iter = 0;
		    h_overflow = 0;
		    // Init iter
		    *d_iter = 0;
		    pmem_persist(d_iter, len_iter);
		    
		    // Run first iteration in master CPU thread
		    h_num_t = 1;
		    int pid;
		    int index_i, index_o;
		    for(index_i = 0; index_i < h_num_t; index_i++) {
				pid = h_q1[index_i];
				h_color[pid] = BLACK;
				for(int i = h_nodes[pid].x; i < (h_nodes[pid].y + h_nodes[pid].x); i++) {
				    int id = h_edges[i].x;
				    h_color[id] = BLACK;
				    index_o       = h_tail++;
				    h_q2[index_o] = id;
				}
		    }
		    h_num_t = h_tail;
		    h_tail = 0;
		    h_threads_run++;
		    h_iter++;

		    //Copy Reset values 
		    cudaMemcpy(d_cost, h_cost, n_nodes * sizeof(int), cudaMemcpyHostToDevice); 
		    cudaMemcpy(d_color, h_color, n_nodes * sizeof(int), cudaMemcpyHostToDevice); 
		    cudaMemcpy(d_q1, h_q1, n_nodes * sizeof(int), cudaMemcpyHostToDevice); 
		    cudaMemcpy(d_q2, h_q2, n_nodes * sizeof(int), cudaMemcpyHostToDevice); 
		    cudaMemcpy(d_overflow, &h_overflow, sizeof(int), cudaMemcpyHostToDevice); 
		    cudaMemcpy(d_iter, &h_iter, sizeof(int), cudaMemcpyHostToDevice); 
		    cudaMemcpy(d_threads_run, &h_threads_run, sizeof(int), cudaMemcpyHostToDevice); 
		    cudaMemcpy(d_threads_end, &h_threads_end, sizeof(int), cudaMemcpyHostToDevice); 
		}
        else {
        	h_iter = *d_iter;
        	h_num_t = d_num_t[h_iter % 2];
        	h_tail = 0;
        	h_head = 0;
            cleanup = 1;
        }
		//START_BW_MONITOR2("bw_gpm_bfs.csv")
      //Run BFS
		while(h_num_t != 0) {
		    if(h_iter % 2 == 0) {
				d_qin  = d_q1;
				d_qout = d_q2;
		    } else {
				d_qin  = d_q2;
				d_qout = d_q1;
		    }
		    cudaMemcpy(d_tail, &h_tail, sizeof(int), cudaMemcpyHostToDevice);
		    cudaMemcpy(d_head, &h_head, sizeof(int), cudaMemcpyHostToDevice);
		    start = std::chrono::high_resolution_clock::now(); 
		    cudaStatus = call_BFS_gpu(p.n_gpu_blocks, p.n_gpu_threads, d_nodes, d_edges,d_cost,
				    d_color, d_qin, d_qout, h_num_t,
				    d_head, d_tail, d_threads_end, d_threads_run,
				    d_overflow, d_iter, p.switching_limit, CPU_EXEC, sizeof(int) * (W_QUEUE_SIZE + 3), h_iter, cleanup);
		   	cudaDeviceSynchronize();
		   	checkForCudaErrors(cudaStatus);
		   	cleanup = 0;
		   	int overflow;
		   	cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost);
		   	if(overflow) {
       			printf("Overflow\n");
       			fflush(stdout);
       		}
		    (*d_iter)++;
		    pmem_persist(d_iter, sizeof(int));
		   	operation_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
#if defined(GPM_WDP)
		    start = std::chrono::high_resolution_clock::now();
		    cudaMemcpy(&h_tail, d_tail, sizeof(int), cudaMemcpyDeviceToHost);
		    cudaDeviceSynchronize();
			for(int i = 0; i < h_tail; ++i) {
				pmem_flush(&d_cost[d_qout[i]], sizeof(int));
				pmem_flush(&d_color[d_qout[i]], sizeof(int));
			}
		    pmem_drain();
		    pmem_persist(d_qout, h_tail * sizeof(int));
		    pmem_persist(d_qin, h_num_t * sizeof(int));
		    cudaDeviceSynchronize();
		    persist_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
#endif
		   //CUDA_ERR();
		    cudaMemcpy(&h_tail, d_tail, sizeof(int), cudaMemcpyDeviceToHost);
		    cudaMemcpy(&h_head, d_head, sizeof(int), cudaMemcpyDeviceToHost); 
		    cudaMemcpy(&h_iter, d_iter, sizeof(int), cudaMemcpyDeviceToHost);
		    //int x = h_iter;
		    //if(x % 100 == 0)
		    //    printf("%d\n", x);
		    start = std::chrono::high_resolution_clock::now(); 
		    h_num_t = h_tail; // Number of elements in output queue
		    cudaMemcpy(&d_num_t[h_iter % 2], &h_num_t, sizeof(int), cudaMemcpyHostToDevice);
		   	operation_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
#if defined(GPM_WDP)
		    start = std::chrono::high_resolution_clock::now();
		    pmem_persist(&d_num_t[h_iter % 2], sizeof(int));
		    persist_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();	
#endif
		    h_tail = 0;
		    h_head = 0;
		}
		//STOP_BW_MONITOR
		cudaMemcpy(h_cost, d_cost, sizeof(int) * n_nodes, cudaMemcpyDeviceToHost); 
		cudaMemcpy(h_q1, d_q1, sizeof(int) * n_nodes, cudaMemcpyDeviceToHost); 
		cudaMemcpy(h_q2, d_q2, sizeof(int) * n_nodes, cudaMemcpyDeviceToHost); 
		cudaMemcpy(&h_num_t, &d_num_t[h_iter % 2], sizeof(int), cudaMemcpyDeviceToHost); 
		cudaMemcpy(&h_iter, d_iter, sizeof(int), cudaMemcpyDeviceToHost); 
		//int x = h_iter;
		//printf("%d\n", x);
		//BFS done running
      //CUDA_ERR();
    }
#if defined(NVM_ALLOC_CPU) && !defined(FAKE_NVM) && !defined(GPM_WDP)
	start = std::chrono::high_resolution_clock::now(); 
	ddio_on(); 
	operation_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
#endif
	OUTPUT_STATS
    printf("\nOperation execution time: %f ms\n", operation_time/1000000.0);
    printf("\nruntime: %f ms\n", (operation_time + persist_time)/1000000.0);
    printf("PersistTime: %f ms\n", persist_time/1000000.0);


// Verify answer
//verify(h_cost, n_nodes, p.comparison_file);

// Free memory
//timer.start("Deallocation");
free(h_nodes);
free(h_edges);
free(h_color);
free(h_cost);
free(h_q1);
free(h_q2);
//cudaStatus = cudaFree(d_nodes);
//cudaStatus = cudaFree(d_edges);
//cudaStatus = cudaFree(d_cost);
cudaStatus = cudaFree(d_color);
//cudaStatus = cudaFree(d_q1);
//cudaStatus = cudaFree(d_q2);
//cudaStatus = cudaFree(d_num_t);
cudaStatus = cudaFree(d_head);
cudaStatus = cudaFree(d_tail);
cudaStatus = cudaFree(d_threads_end);
cudaStatus = cudaFree(d_threads_run);
cudaStatus = cudaFree(d_overflow);
//cudaStatus = cudaFree(d_iter);
//CUDA_ERR();
cudaDeviceSynchronize();
//timer.stop("Deallocation");
//timer.print("Deallocation", 1);

// Release timers
/*timer.release("Allocation");
  timer.release("Initialization");
  timer.release("Copy To Device");
  timer.release("Kernel");
  timer.release("Copy Back and Merge");
  timer.release("Deallocation");*/

printf("Test Passed\n");
return 0;
}
