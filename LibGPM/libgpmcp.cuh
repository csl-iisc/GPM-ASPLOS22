#pragma once
extern "C" 
{
#include "change-ddio.h"
}
#include "libgpm.cuh"
#include <stdio.h>
#include <chrono>
#include <string>

// Non-volatile metadata
struct gpmcp_nv
{
    long long elements;        // Maximum number of elements per group
    long long partitions;      // Number of groups for the cp
    size_t size;         // Total size of data being cp
};

// Volatile metadata
struct gpmcp
{
    const char *path;    // File path
    
    char *index;         // Set of non-volatile indices
    void *start;         // Pointer to start of non-volatile region
    size_t tot_size;     // Total size, including shadow space and metadata
    long long elements;        // Maximum number of elements per group

    gpmcp_nv *cp;        // Pointer to non-volatile metadata

    // Checkpoint entries
    void  **node_addr;   // Set of starting addresses for different elements
    size_t *node_size;   // Set of sizes of each element
    
    // Partition info
    long long *part_byte_size; // Set of cp starting addresses for each group
    long long *part_bytes;     // Size of contents in partition
    long long *part_elem_size; // Set indicating number of elements in each group
};

static __global__ void setup_cp(gpmcp *cp, long long size, long long elements, long long partitions)
{
    cp->cp->elements = elements;
    cp->cp->partitions = partitions;
    cp->cp->size = size;
}

static __global__ void setup_partitions(long long *byte_size, long long partitions, size_t size)
{
    long long ID = threadIdx.x + blockDim.x * blockIdx.x;
    for(long long i = ID; i < partitions; i += gridDim.x * blockDim.x) {
        byte_size[i] = i * size / partitions;
    }
}

static __host__ gpmcp *gpmcp_create(const char *path, size_t size, long long elements, long long partitions)
{
    gpmcp *cp;
    cudaMallocHost((void **)&cp, sizeof(gpmcp));
    cp->path = path;
    
    // Make all blocks of data equal sizes and 128-byte aligned
    // 4-byte alignment improves checkpoint throughput
    size += partitions - (size % partitions > 0 ? size % partitions : partitions);
    // 128 * elements to allow for 128-byte alignment
    size += (4 - (size / partitions % 4 > 0 ? size / partitions % 4 > 0 : 4 ) + 128 * elements) * partitions;
    // 128-byte align size
	size += 128 - (size % 128);

    // Header size + location bitmap
    size_t total_size = sizeof(gpmcp) + partitions; 
    // Aligned 2 * Data size (2xsize for crash redundancy)
    total_size += 128 - total_size % 128 + 2 * size;
        
    cp->tot_size = total_size;
    
    // Map file
#ifdef NVM_ALLOC_GPU
    int is_pmem;
    size_t file_size;
	// Allocate metadata explicitly on CPU as gpm_map will allocate on GPU
    char *cp_pointer = (char *)pmem_map_file(path, total_size - 2 * size, PMEM_FILE_CREATE, 0666, &file_size, &is_pmem);
    printf("Allocated CP metadata %ld size. Pmem? %d\n", file_size, is_pmem);
#else
	// Don't need to worry here, as gpm_map can be accessed from CPU
    char *cp_pointer = (char *)gpm_map_file(path, total_size, 1);
	
#endif

    gpmcp_nv *cp_nv = (gpmcp_nv *)cp_pointer;
    cp_pointer += sizeof(gpmcp_nv);
    cp->cp = cp_nv;
    
    void **node_addr;
    size_t *node_size;
    cudaMallocHost((void **)&node_addr, sizeof(void *) * elements * partitions);
    cudaMallocHost((void **)&node_size, sizeof(size_t) * elements * partitions);
    memset(node_addr, 0, sizeof(void *) * elements * partitions);
    memset(node_size, 0, sizeof(size_t) * elements * partitions);
    
    cp->node_addr = node_addr;
    cp->node_size = node_size;
    cp->index = (char *)cp_pointer;
    cp_pointer += partitions;
    cp_pointer += 128 - (size_t)cp_pointer % 128;
#ifdef NVM_ALLOC_GPU
	size_t cp_size = 2 * size;
	cp->start = (char *)gpm_map_file((std::string(path) + "_gpu").c_str(), cp_size, 1);
#else
    cp->start = cp_pointer;
#endif
    cp->elements = elements;
    
    cudaMallocHost((void **)&cp->part_byte_size, sizeof(long long) * partitions);
    cudaMallocHost((void **)&cp->part_elem_size, sizeof(long long) * partitions);
    cudaMallocHost((void **)&cp->part_bytes, sizeof(long long) * partitions);
    memset(cp->part_elem_size, 0, sizeof(long long) * partitions);
    memset(cp->part_bytes, 0, sizeof(long long) * partitions);
    setup_partitions <<<(partitions + 1023) / 1024, 1024>>>(cp->part_byte_size, partitions, size);
    
    cp->cp->elements = elements;
    cp->cp->partitions = partitions;
    cp->cp->size = size;
    return cp;
}

static __host__ gpmcp *gpmcp_open(const char *path)
{
    gpmcp *cp;
    cudaMallocHost((void **)&cp, sizeof(gpmcp));
    cp->path = path;
    
    size_t len = 0;
    char *cp_pointer = (char *)gpm_map_file(path, len, false);
    gpmcp_nv *cp_nv = (gpmcp_nv *)cp_pointer;
    cp_pointer += sizeof(gpmcp_nv);
    
    cp->tot_size = len;
    cp->cp = cp_nv;
    
    void **node_addr;
    size_t *node_size;
    cudaMallocHost((void **)&node_addr, sizeof(void *) * cp_nv->elements * cp_nv->partitions);
    cudaMallocHost((void **)&node_size, sizeof(size_t) * cp_nv->elements * cp_nv->partitions);
    memset(node_addr, 0, sizeof(void *) * cp_nv->elements * cp_nv->partitions);
    memset(node_size, 0, sizeof(size_t) * cp_nv->elements * cp_nv->partitions);
    
    cp->node_addr = node_addr;
    cp->node_size = node_size;
    cp->index = (char *)cp_pointer;
    cp_pointer += cp_nv->partitions;
    cp_pointer += 128 - (size_t)cp_pointer % 128;
    cp->start = cp_pointer;
    cp->elements = cp_nv->elements;
    
    cudaMallocHost((void **)&cp->part_byte_size, sizeof(long long) * cp_nv->partitions);
    cudaMallocHost((void **)&cp->part_elem_size, sizeof(long long) * cp_nv->partitions);
    cudaMallocHost((void **)&cp->part_bytes, sizeof(long long) * cp_nv->partitions);
    cudaMemset(cp->part_elem_size, 0, sizeof(long long) * cp_nv->partitions);
    cudaMemset(cp->part_bytes, 0, sizeof(long long) * cp_nv->partitions);
    
    setup_partitions <<<(cp_nv->partitions + 1023) / 1024, 1024>>>(cp->part_byte_size, cp_nv->partitions, cp_nv->size);
    return cp;
}

static __host__ void gpmcp_close(gpmcp *cp)
{
    gpm_unmap(cp->cp, cp->tot_size);
    cudaFreeHost(cp->node_addr);
    cudaFreeHost(cp->node_size);
    cudaFreeHost(cp->part_byte_size);
    cudaFreeHost(cp->part_elem_size);
    cudaFreeHost(cp->part_bytes);
    cudaFreeHost(cp);
}

static __host__ long long gpmcp_register(gpmcp *cp, void *addr, size_t size, long long partition)
{
    long long val = 0;
/*#if defined(__CUDA_ARCH__)
    long long start = cp->part_elem_size[partition];
    if(start >= cp->elements)
        return -1;
    // Device code here
    cp->node_addr[start + cp->elements * partition] = (long long *)addr;
    cp->node_size[start + cp->elements * partition] = size;
    cp->part_elem_size[partition]++;
    cp->part_bytes[partition] += size;
    
#else*/
    
    long long start = cp->part_elem_size[partition];
    cp->node_addr[start + cp->elements * partition] = addr;
    cp->node_size[start + cp->elements * partition] = size;
    ++cp->part_elem_size[partition];
    cp->part_bytes[partition] += size;
//#endif
    return val;
}

static __global__ void checkpointKernel(void *start, void *addr, size_t size)
{
    // Find where this thread should write
    size_t offset = (blockDim.x * blockIdx.x + threadIdx.x) * 8;
    
    for(; offset < size; offset += blockDim.x * gridDim.x * 8)
        gpm_memcpy_nodrain((char*)start + offset, 
            (char *)addr + offset, 
            min((size_t)8, size - offset), cudaMemcpyDeviceToHost);
    
    gpm_persist();
}
static __global__ void checkpointKernel_wdp(void *start, void *addr, size_t size)
{
    // Find where this thread should write
    size_t offset = (blockDim.x * blockIdx.x + threadIdx.x) * 8;
    
    for(; offset < size; offset += blockDim.x * gridDim.x * 8)
        gpm_memcpy_nodrain((char*)start + offset, 
            (char *)addr + offset, 
            min((size_t)8, size - offset), cudaMemcpyDeviceToHost);
}


static __host__ long long gpmcp_checkpoint(gpmcp *cp, long long partition)
{
#ifdef GPM_WDP
    char *ptr = getenv("PMEM_THREADS");
    size_t pm_threads;
    if(ptr != NULL)
        pm_threads = atoi(ptr);
    else
        pm_threads = 1;
#endif
    size_t element_offset = 0;
    for(long long i = 0; i < cp->part_elem_size[partition]; ++i) {
		auto start_time = std::chrono::high_resolution_clock::now();
        void *addr = cp->node_addr[i + cp->elements * partition];
        size_t size = cp->node_size[i + cp->elements * partition];
        
        size_t start = cp->part_byte_size[partition] + element_offset;
        
        // Based on index move to working copy
        char ind = (cp->index[partition] != 0 ? 0 : 1);
        start += ind * cp->cp->size;
                
        // Host code
        const long long threads = 1024;
        long long blocks = 1;
        // Have each threadblock persist a single element
        // Threads within a threadblock persist at 4-byte offsets
#ifdef NVM_ALLOC_GPU
        checkpointKernel<<<120, threads>>>((void*)((char*)cp->start + start), addr, size);
#else
		checkpointKernel<<<blocks, threads>>>((void*)((char*)cp->start + start), addr, size);
#endif 
    	cudaDeviceSynchronize();
#ifdef CHECKPOINT_TIME
    	checkpoint_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
#endif
#ifdef GPM_WDP
		start_time = std::chrono::high_resolution_clock::now();
        size_t GRAN = (size + pm_threads - 1) / pm_threads;
        #pragma omp parallel for num_threads(pm_threads)
        for(size_t ind = 0; ind < pm_threads; ++ind)
            pmem_persist((void*)((char*)cp->start + start + ind * GRAN),  min(GRAN, size - ind * GRAN)); 
#ifdef PERSIST_TIME
		persist_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
#endif
#endif 
        element_offset += size + 128 - (size % 128);
    }
	auto start_time = std::chrono::high_resolution_clock::now();
    // Update index
    cp->index[partition] ^= 1;
    pmem_persist(&cp->index[partition], sizeof(cp->index[partition]));
#ifdef CHECKPOINT_TIME
	checkpoint_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
#endif
    return 0;
//#endif
//#endif
}

static __device__ long long gpmcp_checkpoint_start(gpmcp *cp, long long partition, long long element, size_t offset, size_t size)
{
    // Device code
    size_t start = cp->part_byte_size[partition];
    
    char ind = cp->index[partition];
    ind = (ind != 0 ? 0 : 1);
    size_t cp_size = cp->cp->size;
    start += ind * cp_size;
    
    long long elems = cp->cp->elements;
    for(long long i = 0; i < element; ++i)
        start += cp->node_size[partition * elems + i];
    
    if(start >= 2 * cp_size)
        return -1;

    void *addr = (char *)cp->node_addr[partition * elems + element] + offset;
    gpm_memcpy((char *)cp->start + start + offset, addr, size, cudaMemcpyDeviceToDevice);
    return 0;
}

static __device__ long long gpmcp_checkpoint_value(gpmcp *cp, long long partition, long long element, size_t offset, size_t size, void *addr)
{
    // Device code
    size_t start = cp->part_byte_size[partition];
    
    char ind = cp->index[partition];
    ind = (ind != 0 ? 0 : 1);
    size_t cp_size = cp->cp->size;
    start += ind * cp_size;
    
    long long elems = cp->cp->elements;
    for(long long i = 0; i < element; ++i)
        start += cp->node_size[partition * elems + i];
    
    if(start >= 2 * cp_size)
        return -1;

    gpm_memcpy((char *)cp->start + start + offset, addr, size, cudaMemcpyDeviceToDevice);
    return 0;
}

static __device__ long long gpmcp_checkpoint_finish(gpmcp *cp, long long partition)
{
    // Update index once complete
    cp->index[partition] ^= 1;
    gpm_persist();
    return 0;
}

__global__ void restoreKernel(void *start, void *addr, size_t size)
{
    // Find where this thread should write
    size_t offset = (blockDim.x * blockIdx.x + threadIdx.x) * 8;
    
    for(; offset < size; offset += blockDim.x * gridDim.x * 8)
        gpm_memcpy_nodrain((char*)start + offset, 
            (char *)addr + offset, 
            min((size_t)8, size - offset), cudaMemcpyDeviceToHost);
}

static __host__ long long gpmcp_restore(gpmcp *cp, long long partition)
{
    size_t element_offset = 0;
    for(long long i = 0; i < cp->part_elem_size[partition]; ++i) {
        void *addr = cp->node_addr[i + cp->elements * partition];
        size_t size = cp->node_size[i + cp->elements * partition];
        
        size_t start = cp->part_byte_size[partition] + element_offset;
        
        // Based on index move to working copy
        char ind = (cp->index[partition] != 0 ? 1 : 0);
        start += ind * cp->cp->size;
                
        // Host code
        const long long threads = 1024;
        long long blocks = 1;
        // Have each threadblock persist a single element
        // Threads within a threadblock persist at 4-byte offsets
        restoreKernel<<<blocks, threads>>>(addr, (void*)((char*)cp->start + start), size);
        element_offset += size + 128 - (size % 128);
    }
	cudaDeviceSynchronize();
    return 0;
}
