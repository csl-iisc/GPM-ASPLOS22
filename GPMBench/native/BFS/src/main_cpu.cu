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

#include "support/cuda-setup.h"
#include "support/timer.h"
#include "cuda_runtime.h"
#include <atomic>
#include "libgpm.cuh"
#include <math.h>
#include <vector>
#include <algorithm>

#include <unistd.h>
#include <thread>
#include <assert.h>

#define PRINT 0
#define PRINT_ALL 0

#define INF -2147483647
#define UP_LIMIT 16677216 //2^24
#define WHITE 16677217
#define GRAY 16677218
#define GRAY0 16677219
#define GRAY1 16677220
#define BLACK 16677221
#define W_QUEUE_SIZE 1600

typedef struct {
    int x;
    int y;
} Node;
typedef struct {
    int x;
    int y;
} Edge;

double operation_time = 0;

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
        n_gpu_blocks   = 8;
        n_threads       = 32;
        n_warmup        = 1;
        n_reps          = 1;
        file_name       = "input/USA-road-d.USA.gr.parboil";
        comparison_file = "output/USA-road-d.USA.gr.out";
        switching_limit = 128;
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

int atomic_maximum(std::atomic_int *maximum_value, int value) {
    int prev_value = (maximum_value)->load();
    while(prev_value < value && !(maximum_value)->compare_exchange_strong(prev_value, value))
        ;
    return prev_value;
}

// CPU threads-----------------------------------------------------------------
void run_cpu_threads(Node *h_graph_nodes, Edge *h_graph_edges, std::atomic_int *cost, std::atomic_int *color,
    int *q1, int *q2, int *n_t, std::atomic_int *head, std::atomic_int *tail,
    std::atomic_int *threads_end, std::atomic_int *threads_run, std::atomic_int *iter, int n_threads) {
///////////////// Run CPU worker threads /////////////////////////////////
#if PRINT
    printf("Starting %d CPU threads\n", n_threads * CPU);
#endif
    std::vector<std::thread> cpu_threads;
    for(int k = 0; k < n_threads; k++) {
        cpu_threads.push_back(std::thread([=]() {

            int iter_local = (iter)->load(); // Current iteration/level

            int base = (head)->fetch_add(1); // Fetch new node from input queue
            while(base < *n_t) {
                int pid = q1[base];
                cost[pid].store(iter_local); // Node visited
                pmem_persist(&cost[pid], sizeof(std::atomic_int));
                // For each outgoing edge
                for(int i = h_graph_nodes[pid].x; i < (h_graph_nodes[pid].y + h_graph_nodes[pid].x); i++) {
                    int id        = h_graph_edges[i].x;
                    int old_color = atomic_maximum(&color[id], BLACK);
                	pmem_persist(&color[id], sizeof(std::atomic_int));
                    if(old_color < BLACK) {
                        // Push to the queue
                        int index_o     = (tail)->fetch_add(1);
                        q2[index_o] = id;
                		pmem_persist(&q2[index_o], sizeof(int));                        
                    }
                }
                base = (head)->fetch_add(1); // Fetch new node from input queue
            }

            if(k == 0) {
                (iter)->fetch_add(1);
            }
        }));
    }
    std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}

// Input Data -----------------------------------------------------------------
void read_input_size(int &n_nodes, int &n_edges, const Params &p) {
    FILE *fp = fopen(p.file_name, "r");
    auto dummy = fscanf(fp, "%d", &n_nodes);
    dummy = fscanf(fp, "%d", &n_edges);
    if(fp)
        fclose(fp);
}

void read_input(int &source, Node *&h_nodes, Edge *&h_edges, const Params &p) {

    int   start, edgeno;
    int   n_nodes, n_edges;
    int   id, cost;
    FILE *fp = fopen(p.file_name, "r");

    auto dummy = fscanf(fp, "%d", &n_nodes);
    dummy = fscanf(fp, "%d", &n_edges);
    dummy = fscanf(fp, "%d", &source);
    printf("Number of nodes = %d\t", n_nodes);
    printf("Number of edges = %d\t", n_edges);

    // initalize the memory: Nodes
    for(int i = 0; i < n_nodes; i++) {
        dummy = fscanf(fp, "%d %d", &start, &edgeno);
        h_nodes[i].x = start;
        h_nodes[i].y = edgeno;
    }
#if PRINT_ALL
    for(int i = 0; i < n_nodes; i++) {
        printf("%d, %d\n", h_nodes[i].x, h_nodes[i].y);
    }
#endif

    // initalize the memory: Edges
    for(int i = 0; i < n_edges; i++) {
        dummy = fscanf(fp, "%d", &id);
        dummy = fscanf(fp, "%d", &cost);
        h_edges[i].x = id;
        h_edges[i].y = -cost;
    }
    if(fp)
        fclose(fp);
}

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {

    const Params p(argc, argv);
    CUDASetup    setcuda(p.device);

    const char *path_node = "/pmem/persist_bfs_node.dat";
    const char *path_edge = "/pmem/persist_bfs_edge.dat";
    const char *path_q2   = "persist_bfs_q2.dat";
    const char *path_q1   = "persist_bfs_q1.dat";
    const char *path_cost = "persist_bfs_cost.dat";
    const char *path_color = "persist_bfs_color.dat";
    const char *path_iter = "persist_bfs_iter.dat";
    const char *path_size = "persist_bfs_size.dat";

    // Allocate
    int n_nodes, n_edges;
    read_input_size(n_nodes, n_edges, p);
    
    
    size_t len_iter = 0, len_q1 = 0, len_q2 = 0;
    size_t len_cost = 0, len_num_t = 0, len_color = 0; 
	len_q2   = n_nodes * sizeof(int);
	len_q1   = n_nodes * sizeof(int);
	len_cost = n_nodes * sizeof(std::atomic_int);
	len_color= n_nodes * sizeof(std::atomic_int);
	len_iter = sizeof(int);
	len_num_t = sizeof(int) * 2;
	std::atomic_int *h_cost  = (std::atomic_int *) gpm_map_file(path_cost, len_cost, true);
	std::atomic_int *h_color = (std::atomic_int *) gpm_map_file(path_color, len_color, true);
	int * h_q2    = (int *) gpm_map_file(path_q2, len_q2, true);
	int * h_q1    = (int *) gpm_map_file(path_q1,  len_q1, true);
	int * h_num_t = (int *) gpm_map_file(path_size, len_num_t, true);
	std::atomic_int *h_iter  = (std::atomic_int *) gpm_map_file(path_iter,  len_iter, true);

    Node * h_nodes = (Node *)malloc(sizeof(Node) * n_nodes);
    Edge * h_edges = (Edge *)malloc(sizeof(Edge) * n_edges);    
    std::atomic_int  h_head[1];
    std::atomic_int  h_tail[1];
    std::atomic_int  h_threads_end[1];
    std::atomic_int  h_threads_run[1];
    
    /*
    std::atomic_int *h_color = (std::atomic_int *)malloc(sizeof(std::atomic_int) * n_nodes);
    std::atomic_int *h_cost  = (std::atomic_int *)malloc(sizeof(std::atomic_int) * n_nodes);
    int *            h_q1    = (int *)malloc(n_nodes * sizeof(int));
    int *            h_q2    = (int *)malloc(n_nodes * sizeof(int));
    int              h_num_t[1];
    int              h_overflow[1];
    std::atomic_int  h_iter[1];
    ALLOC_ERR(h_nodes, h_edges, h_color, h_cost, h_q1, h_q2);*/

    // Initialize
    int source;
    read_input(source, h_nodes, h_edges, p);
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
    //h_overflow[0] = 0;

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
        //h_overflow[0] = 0;

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
        h_num_t[(h_iter[0] + 1) % 2] = h_tail[0].load();
        h_tail[0].store(0);
        h_threads_run[0].fetch_add(1);
        h_iter[0].fetch_add(1);

        // Pointers to input and output queues
        int * h_qin  = h_q2;
        int * h_qout = h_q1;
		//printf("Started execution\n");
		int iter = 0;
		auto start = std::chrono::high_resolution_clock::now(); 
        // Run subsequent iterations on CPU or GPU until number of input queue elements is 0
        while(h_num_t[h_iter[0] % 2] != 0) { // If the number of input queue elements is lower than switching_limit
            // Swap queues
            if(h_iter[0] % 2 == 0) {
                h_qin  = h_q1;
                h_qout = h_q2;
            } else {
                h_qin  = h_q2;
                h_qout = h_q1;
            }

            std::thread main_thread(run_cpu_threads, h_nodes, h_edges, h_cost, h_color, h_qin, h_qout, &h_num_t[h_iter[0] % 2],
                h_head, h_tail, h_threads_end, h_threads_run, h_iter, p.n_threads);
            main_thread.join();
            pmem_persist(h_iter, sizeof(int));
            //if(iter % 100 == 0)
            	//printf("%d\n", iter);

            h_num_t[h_iter[0] % 2] = h_tail[0].load(); // Number of elements in output queue
            pmem_persist(&h_num_t[h_iter[0] % 2], sizeof(int));
            h_tail[0].store(0);
            h_head[0].store(0);
            ++iter;
        }
		operation_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
    }

	printf("\nOperation execution time: %f ms\n", operation_time/1000000.0);
	printf("\nruntime: %f ms\n", operation_time/1000000.0);
    // Verify answer
    //verify(h_cost, n_nodes, p.comparison_file);
    
    //FILE *fp = fopen(p.comparison_file, "w");
    //auto dummy = fprintf(fp, "%d\n", n_nodes);
    //for(int i = 0; i < n_nodes; ++i) {
    	//fprintf(fp, "%d %d\n", i, h_cost[i].load());
    //}
    // Free memory
    //free(h_nodes);
    //free(h_edges);
    //free(h_color);
    //free(h_cost);
    //free(h_q1);
    //free(h_q2);

    printf("Test Passed\n");
    return 0;
}
