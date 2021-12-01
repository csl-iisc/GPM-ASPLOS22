#pragma once
#include "libgpm.cuh"
#include <stdio.h>
#include <time.h>
#include <iostream>
using namespace std;
enum gpmlog_flags
{
    PMEMLOG_UNMANAGED = 1, // User manually specifies where to insert
    PMEMLOG_UNSTRICT  = 2, // Do not persist after every insert
};

// Persistent metadata to be stored in GPU
struct gpu_gpmlog
{
    int partitions;
    /* For non-managed gpmlogs, head/tail
     * stores an index per partition.
     * For managed, it stores one per thread */
    size_t *head;
    size_t *tail;
    size_t byte_size;
    int flags;
};

// Other volatile metadata used during execution
struct gpmlog
{
    // Careful! Below is a host pointer
    // while the other pointers are device
    const char *path;
    size_t log_size; // non-volatile metadata size + byte_size
    struct gpu_gpmlog *plog;
    void *start;
    int *locks; // Used to serialize access when necessary
};

// There should be a CUDA function to
// calculate warp size, but in meantime...
#define WARP_SIZE 32
#define BLOCK_SIZE 128
#define WORD_SIZE 4

/************************
 *
 *  INTERNAL FUNCTIONS
 *
 ************************/
static __global__ void setup_log(gpmlog *plog, char *start, int len)
{
    char *start_copy = start;
    
    size_t num_heads = ((size_t)plog->plog->tail - (size_t)plog->plog->head) / sizeof(size_t);
    plog->plog->head = (size_t *)start;
    start = (char *)start + sizeof(size_t) * num_heads;
    plog->plog->tail = (size_t *)start;
    start = (char *)start + sizeof(size_t) * num_heads;
    
    if(!(plog->plog->flags & PMEMLOG_UNMANAGED))
    {
        size_t header_size = (size_t)start - (size_t)start_copy + sizeof(gpu_gpmlog);
        start = (char *)start + (header_size % BLOCK_SIZE != 0 ? BLOCK_SIZE - header_size % BLOCK_SIZE : 0);
    }
    
    plog->start = (char *)start;
}

static __device__ int getGlobalIdx()
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
        
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x) + threadIdx.x;
        
    return threadId;
}

static __device__ int gpmlog_insert_manual(gpmlog *plog, void *var, size_t size, int partition)
{
    if(partition >= plog->plog->partitions) {
        return -1;
    }
    
    bool done = false;
    do {
        // Lock partition
        int locked = atomicCAS(&plog->locks[partition], 0, 1);
        __threadfence();
        if(!locked) {
            done = true;
            PMEM_READ_OP( size_t start = plog->plog->head[partition] , 8 )
            PMEM_READ_OP(  , 8 )
            if(start + size > plog->plog->tail[partition]) {
                size = plog->plog->tail[partition] - start;
            }
            
            
            PMEM_READ_OP( , 4 )
            // Insert into log
            // Only update head after all memory has 
            // been placed to maintain crash consistency
            if(plog->plog->flags & PMEMLOG_UNSTRICT) {
                PMEM_READ_OP( gpm_memcpy_nodrain((char *)plog->start + start, var, size, cudaMemcpyDeviceToDevice) , size )
                PMEM_READ_OP( size_t temp = plog->plog->head[partition] + size , 8 )
                gpm_memcpy_nodrain((char *)&plog->plog->head[partition], &temp, sizeof(size_t), cudaMemcpyDeviceToDevice);
            }
            else {
                PMEM_READ_OP( gpm_memcpy((char *)plog->start + start, var, size, cudaMemcpyDeviceToDevice) , size )
                PMEM_READ_OP( size_t temp = plog->plog->head[partition] + size , 8 )
                gpm_memcpy((char *)&plog->plog->head[partition], &temp, sizeof(size_t), cudaMemcpyDeviceToDevice);
            }
            __threadfence();
            atomicExch(&plog->locks[partition], 0);
        }
    } while(!done);
    
    return size;
}

static __device__ int gpmlog_insert_managed(gpmlog *plog, void *var, size_t size, int partition)
{    
    int tid;
    if(partition == -1)
        tid = getGlobalIdx();
    else
        tid = partition;
    PMEM_READ_OP( size_t temp_head = plog->plog->head[tid] , sizeof(size_t) )
    //printf("Temp head: %lu\n", temp_head);
    PMEM_READ_OP( size_t tail = plog->plog->tail[tid] , sizeof(size_t) )
    int i = 0;
    for(; i < size && temp_head < tail;) {
        // Write up to a word at a time
        int sz = 1;
        if(size - i >= WORD_SIZE - (int)(temp_head % WORD_SIZE))
            sz = WORD_SIZE - (int)(temp_head % WORD_SIZE);
        
        PMEM_READ_OP( gpm_memcpy_nodrain((char *)plog->start + temp_head, (char *)var + i, sz, cudaMemcpyDeviceToDevice) , sz )
        
        temp_head += (size_t)sz;
        i += sz;
        // Reached end of WORD_SIZE byte segment
        // move to next segment
        if(temp_head % WORD_SIZE == 0) {
            // next address = current address + BLOCK_SIZE - WORD_SIZE
            temp_head += BLOCK_SIZE - WORD_SIZE;
        }
    }
    PMEM_READ_OP( , 8 )
    if(i < size && temp_head >= plog->plog->tail[tid]) {
        size = i;
    }
    
    PMEM_READ_OP( , 4 )
    // Only update head after all memory has 
    // been placed to maintain crash consistency
    if(PMEMLOG_UNSTRICT & plog->plog->flags) {
        gpm_memcpy_nodrain(&plog->plog->head[tid], &temp_head, sizeof(size_t), cudaMemcpyDeviceToDevice);
    }
    else {
        gpm_drain();
        gpm_memcpy(&plog->plog->head[tid], &temp_head, sizeof(size_t), cudaMemcpyDeviceToDevice);
    }
    return i;
}

static __device__ int gpmlog_read_manual(gpmlog *plog, void *var, size_t size, int partition)
{
    if(partition >= plog->plog->partitions) {
        return -1;
    }

    bool done = false;
    do {
        // Lock partition
        int locked = atomicCAS(&plog->locks[partition], 0, 1);
        __threadfence();
        if(!locked) {
            done = true;
            size_t start = plog->plog->head[partition];
            
            if((partition == 0 && start < size) || (partition != 0 && start < plog->plog->tail[partition - 1] + size)) {
                if(partition == 0)
                    size = start;
                else
                    size = start - plog->plog->tail[partition - 1];
            }
            
            // Read data stored in log
            vol_memcpy(var, (char *)plog->start + start - size, size);
            
            // Unlock
            __threadfence();
            atomicExch(&plog->locks[partition], 0);
        }
    } while(!done);
    return size;
}

static __device__ int gpmlog_read_managed(gpmlog *plog, void *var, size_t size, int partition)
{    
    int tid = getGlobalIdx();
    if(partition != -1)
        tid = partition;
    
    // Check for underflow condition
    size_t head = plog->plog->head[tid] - (tid < WARP_SIZE ? 0 : plog->plog->tail[tid - WARP_SIZE]);
    if(head / BLOCK_SIZE * WORD_SIZE + head % WORD_SIZE < size) {
        size = head / BLOCK_SIZE * WORD_SIZE + head % WORD_SIZE;
    }
    
    size_t i = plog->plog->head[tid], j = size;
    for(;j > 0;) {
        // Reached end of WORD_SIZE byte segment
        // move to previous segment
        if(i % WORD_SIZE == 0) {
            // next address = current address - BLOCK_SIZE + WORD_SIZE
            i -= BLOCK_SIZE - WORD_SIZE;
        }
        
        // Read up to a word at a time
        size_t sz = 1;
        if(i % WORD_SIZE == 0 && j >= WORD_SIZE)
            sz = WORD_SIZE;
        else if(j >= i % WORD_SIZE)
            sz = i % WORD_SIZE;
        else
            sz = j;
        
        i -= sz;
        j -= sz;
        vol_memcpy((char *)var + j, (char *)plog->start + i, sz);
    }
    return size;
}

static __device__ int gpmlog_remove_manual(gpmlog *plog, size_t size, int partition)
{
    if(partition >= plog->plog->partitions) {
        return -1;
    }
    
    bool done = false;
    do {
        // Lock partition
        int locked = atomicCAS(&plog->locks[partition], 0, 1);
        __threadfence();
        if(!locked) {
            done = true;
            size_t start = plog->plog->head[partition];
            
            if((partition == 0 && start < size) || (partition != 0 && start < plog->plog->tail[partition - 1] + size)) {
                if(partition == 0)
                    size = start;
                else
                    size = start - plog->plog->tail[partition - 1];
            }
            
            size_t temp = plog->plog->head[partition] - size;
            // Drain if necessary
            if(PMEMLOG_UNSTRICT & plog->plog->flags) {   
                gpm_memcpy_nodrain(&plog->plog->head[partition], &temp, sizeof(size_t), cudaMemcpyDeviceToDevice);
            }
            else {
                gpm_memcpy(&plog->plog->head[partition], &temp, sizeof(size_t), cudaMemcpyDeviceToDevice);
            }
            
            // Unlock
            __threadfence();
            atomicExch(&plog->locks[partition], 0);
        }
    } while(!done);
    
    return size;
}

static __device__ int gpmlog_remove_managed(gpmlog *plog, size_t size)
{    
    int tid = getGlobalIdx();
    
    // Check for underflow condition
    size_t head = plog->plog->head[tid] - (tid < WARP_SIZE ? 0 : plog->plog->tail[tid - WARP_SIZE]);
    if(head / BLOCK_SIZE * WORD_SIZE + head % WORD_SIZE < size) {
        size = head / BLOCK_SIZE * WORD_SIZE + head % WORD_SIZE;
    }
    
    // Calculate size of stack after removal
    int new_head = head / BLOCK_SIZE * WORD_SIZE + head % WORD_SIZE - size;
    // Move head to appropriate position
    int offset = (tid < WARP_SIZE ? 0 : plog->plog->tail[tid - WARP_SIZE]) + (tid % WARP_SIZE) * WORD_SIZE;
    new_head = new_head / WORD_SIZE * BLOCK_SIZE + new_head % WORD_SIZE + offset;
    // Drain if necessary
    if(PMEMLOG_UNSTRICT & plog->plog->flags) {   
        gpm_memcpy_nodrain(&plog->plog->head[tid], &new_head, sizeof(size_t), cudaMemcpyDeviceToDevice);
    }
    else {
        gpm_memcpy(&plog->plog->head[tid], &new_head, sizeof(size_t), cudaMemcpyDeviceToDevice);
    }
    return size;
}

static __device__ void gpmlog_clear_manual(gpmlog *plog, int partition)
{
    bool done = false;
    do {
        // Lock partition
        int locked = atomicCAS(&plog->locks[partition], 0, 1);
        if(!locked) {
            done = true;

            // Update log head
            if(partition == 0)
                gpm_memset_nodrain(&plog->plog->head[partition], 0, sizeof(size_t));
            else {
                PMEM_READ_OP( , sizeof(size_t) )
                gpm_memcpy_nodrain(&plog->plog->head[partition], &plog->plog->tail[partition - 1], sizeof(size_t), cudaMemcpyDeviceToDevice);
            }
            // Drain if necessary
            if(!(PMEMLOG_UNSTRICT & plog->plog->flags))
                gpm_drain();

            atomicExch(&plog->locks[partition], 0);
        }
    } while(!done);
}

static __device__ void gpmlog_clear_managed(gpmlog *plog, int partition)
{
    int tid = getGlobalIdx();
    if(partition != -1)
        tid = partition;
    PMEM_READ_OP( , sizeof(size_t) )
    size_t head = (tid % WARP_SIZE) * WORD_SIZE + (tid < WARP_SIZE ? 0 : plog->plog->tail[tid - WARP_SIZE]);
    
    gpm_memcpy_nodrain(&plog->plog->head[tid], &head, sizeof(size_t), cudaMemcpyDeviceToDevice);
    
    // Drain if necessary
    if(!(PMEMLOG_UNSTRICT & plog->plog->flags))
        gpm_drain();
}

static __global__ void setupPartitions(size_t *head, size_t *tail, int partitions, size_t len)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= partitions)
        return;
    
    head[id] = (len / partitions) * id;
    tail[id] = (len / partitions) * (id + 1);
    if(id == partitions - 1)
        tail[id] = len;
}

static __host__ gpmlog *gpmlog_create(const char *path, size_t len, int partitions, int flags = 0)
{
    // Mark as unmanaged
    flags |= PMEMLOG_UNMANAGED;
    
    // Create volatile metadata
    gpmlog *plog, dummy_log;
    // TODO: make this cudaMalloc
    cudaMalloc((void **)&plog, sizeof(gpmlog));
    dummy_log.path = path;
    
    // Calculate log size = size of metadata + (heads + tails) + actual log
    dummy_log.log_size = sizeof(gpu_gpmlog) + (2 * sizeof(size_t) * partitions) + len;
    
    // Create persistent memory chunk for persistent part
    void *log_pointer = gpm_map_file(path, dummy_log.log_size, 1);
    int *locks;
    cudaMalloc((void **)&locks, sizeof(int) * partitions);
    cudaMemset(locks, 0, sizeof(int) * partitions);
    
    // Set appropriate pointer locations
    dummy_log.plog = (gpu_gpmlog *)log_pointer;
    dummy_log.locks = locks;
    
    // Shift appropriate amount for next pointer
    log_pointer = (char *)log_pointer + sizeof(gpu_gpmlog);
    
    // Assign values for persistent log metadata
    gpu_gpmlog temp_log;
    temp_log.partitions = partitions;
    temp_log.head       = (size_t *)log_pointer;
    log_pointer = (char *)log_pointer + sizeof(size_t) * partitions;
    temp_log.tail       = (size_t *)log_pointer;
    temp_log.byte_size  = len;
    temp_log.flags      = flags;
    
    log_pointer = (char *)log_pointer + sizeof(size_t) * partitions;
    dummy_log.start = log_pointer;
    
    cudaMemcpy(dummy_log.plog, &temp_log, sizeof(gpu_gpmlog), cudaMemcpyHostToDevice);
    
    setupPartitions<<<(partitions + 511) / 512, 512>>> (temp_log.head, temp_log.tail, partitions, len);
    cudaMemcpy(plog, &dummy_log, sizeof(gpmlog), cudaMemcpyHostToDevice);
    return plog;
}

static __global__ void setupPartitionsManaged(size_t *head, size_t *tail, int blocks, int threads, size_t len)
{
    int num_warps = (blocks * ((threads + WARP_SIZE - 1) / WARP_SIZE));
    int blk = blockIdx.x;
    int thd = threadIdx.x;
    if(blk >= blocks || thd >= threads)
        return;
    
    size_t regions_per_warp = len / (BLOCK_SIZE * num_warps);
    int id = blk * threads + thd;
    
    head[blk * threads + thd] = (regions_per_warp * BLOCK_SIZE) * (id / 32) + (id % 32) * WORD_SIZE;
    tail[blk * threads + thd] = (regions_per_warp * BLOCK_SIZE) * (id / 32 + 1);
    
    if(blk == blocks - 1 && thd >= threads - 32)
        tail[blk * threads + thd] = len;
}


static __host__ gpmlog *gpmlog_create_managed(const char *path, size_t &len, int blocks, int threads, int flags = 0)
{
    gpmlog *plog, dummy_log;
    // TODO: make this cudaMalloc
    cudaMalloc((void **)&plog, sizeof(gpmlog));
    dummy_log.path = path;
    
    size_t extra_len = len / (blocks * threads);
    extra_len = (extra_len + 3) / 4;
    // Convert len into BLOCK_SIZE byte blocks
    len = (len + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    int num_warps = (blocks * ((threads + WARP_SIZE - 1) / WARP_SIZE));
    if(len < BLOCK_SIZE * num_warps * extra_len)
        len = BLOCK_SIZE * num_warps * extra_len;
    //printf("Len: %lu\n", len); 
    size_t header_size = sizeof(gpu_gpmlog) + 2 * sizeof(size_t) * blocks * threads;
    // Calculate log size = size of metadata + (heads + tails) + offset + actual log (starting at BLOCK_SIZE byte offset)
    dummy_log.log_size = header_size + (header_size % BLOCK_SIZE != 0 ? BLOCK_SIZE - header_size % BLOCK_SIZE : 0) + len;
    // Create persistent memory chunk for persistent part
    void *log_pointer = gpm_map_file(path, dummy_log.log_size, 1);
    
    // Set appropriate pointer locations
    dummy_log.plog = (gpu_gpmlog *)log_pointer;
    
    // Shift appropriate amount for next pointer
    log_pointer = (char *)log_pointer + sizeof(gpu_gpmlog);
    
    // Assign values for persistent log metadata
    gpu_gpmlog temp_log;
    temp_log.partitions = blocks * threads;
    temp_log.head       = (size_t *)log_pointer;
    log_pointer = (char *)log_pointer + sizeof(size_t) * blocks * threads;
    temp_log.tail       = (size_t *)log_pointer;
    temp_log.byte_size  = len;
    temp_log.flags      = flags;
    
    log_pointer = (char *)log_pointer + sizeof(size_t) * blocks * threads + 
        (header_size % BLOCK_SIZE != 0 ? BLOCK_SIZE - header_size % BLOCK_SIZE : 0);
    dummy_log.start = log_pointer;
    
    cudaMemcpy(dummy_log.plog, &temp_log, sizeof(gpu_gpmlog), cudaMemcpyHostToDevice);
    setupPartitionsManaged<<<blocks, threads>>> (temp_log.head, temp_log.tail, blocks, threads, len);
    cudaMemcpy(plog, &dummy_log, sizeof(gpmlog), cudaMemcpyHostToDevice);

    return plog;
}

static __host__ gpmlog *gpmlog_open(const char *path)
{
    size_t len = 0;
    char *start = (char *)gpm_map_file(path, len, false);
    // Create volatile metadata
    gpmlog *plog, *real_plog;
    // TODO: make this cudaMalloc
    cudaMallocHost((void **)&plog, sizeof(gpmlog));
    plog->path = path;
    plog->plog = (gpu_gpmlog *)start;
    plog->log_size = len;
    start = start + sizeof(gpu_gpmlog);
    setup_log<<<1, 1>>>(plog, start, len);
    cudaDeviceSynchronize();
    
    // Create locks if necessary
    gpu_gpmlog *temp = new gpu_gpmlog;
    cudaMemcpy(temp, plog->plog, sizeof(gpu_gpmlog), cudaMemcpyDeviceToHost);
    if(temp->flags & PMEMLOG_UNMANAGED)
    {
        cudaMalloc((void **)&plog->locks, sizeof(int) * temp->partitions);
        cudaMemset(plog->locks, 0, sizeof(int) * temp->partitions);
    }
    cudaMalloc((void**)&real_plog, sizeof(gpmlog));
    cudaMemcpy(real_plog, plog, sizeof(gpmlog), cudaMemcpyHostToDevice);
    cudaFreeHost(plog);
    
    return real_plog;
}

static __host__ void gpmlog_close(gpmlog *plog)
{
    gpmlog dummy;
    cudaMemcpy(&dummy, plog, sizeof(gpmlog), cudaMemcpyDeviceToHost);
    gpm_unmap(dummy.path, dummy.plog, dummy.log_size);
    cudaFree(plog);
}

static __device__ int gpmlog_insert(gpmlog *plog, void *var, size_t size, int partition = -1)
{
    if(plog->plog->flags & PMEMLOG_UNMANAGED) {
        if(partition == -1) {
            return -1;
        }
        return gpmlog_insert_manual(plog, var, size, partition);
    }
    else {
        return gpmlog_insert_managed(plog, var, size, partition);
    }
}

static __device__ int gpmlog_read(gpmlog *plog, void *var, size_t size, int partition = -1)
{
    if(plog->plog->flags & PMEMLOG_UNMANAGED) {
        if(partition == -1) {
            return -1;
        }
        return gpmlog_read_manual(plog, var, size, partition);
    }
    else {
        return gpmlog_read_managed(plog, var, size, partition);
    }
}

static __device__ int gpmlog_remove(gpmlog *plog, size_t size, int partition = -1)
{
    if(plog->plog->flags & PMEMLOG_UNMANAGED) {
        if(partition == -1) {
            return -1;
        }
        return gpmlog_remove_manual(plog, size, partition);
    }
    else {
        return gpmlog_remove_managed(plog, size);
    }
}


static __device__ int gpmlog_read_remove(gpmlog *plog, void *var, size_t size, int partition = -1)
{
    if(plog->plog->flags & PMEMLOG_UNMANAGED) {
        if(partition == -1) {
            return -1;
        }
        
        if(partition >= plog->plog->partitions) {
            return -1;
        }
        bool done = false;
        do {
            // Lock partition
            int locked = atomicCAS(&plog->locks[partition], 0, 1);
            __threadfence();
            if(!locked) {
                done = true;
                size_t start = plog->plog->head[partition];
                
                if((partition == 0 && start < size) || (partition != 0 && start < plog->plog->tail[partition - 1] + size)) {
                    if(partition == 0)
                        size = start;
                    else
                        size = start - plog->plog->tail[partition - 1];
                }
                
                // Read data stored in log
                vol_memcpy(var, (char *)plog->start + start - size, size);
                
                // Reduce head to required size
                size_t temp = plog->plog->head[partition] - size;
                if(PMEMLOG_UNSTRICT & plog->plog->flags) {   
                    gpm_memcpy_nodrain(&plog->plog->head[partition], &temp, sizeof(size_t), cudaMemcpyDeviceToDevice);
                }
                else {
                    gpm_memcpy(&plog->plog->head[partition], &temp, sizeof(size_t), cudaMemcpyDeviceToDevice);
                }
                
                // Unlock
                __threadfence();
                atomicExch(&plog->locks[partition], 0);
            }
        } while(!done);
        return size;
    }
    else {
        int tid = getGlobalIdx();
        if(partition != -1)
            tid = partition;
        // Check for underflow condition
        size_t head = plog->plog->head[tid] - (tid < WARP_SIZE ? 0 : plog->plog->tail[tid - WARP_SIZE]);
        if(head / BLOCK_SIZE * WORD_SIZE + head % WORD_SIZE < size) {
            size = head / BLOCK_SIZE * WORD_SIZE + head % WORD_SIZE;
        }
        
        int i = plog->plog->head[tid], j = size;
        for(;j > 0;) {
            // Reached end of WORD_SIZE byte segment
            // move to previous segment
            if(i % WORD_SIZE == 0) {
                // next address = current address - BLOCK_SIZE + WORD_SIZE
                i -= BLOCK_SIZE - WORD_SIZE;
            }
            
            // Read up to a word at a time
            int sz = 1;
            if(i % WORD_SIZE == 0 && j >= WORD_SIZE)
                sz = WORD_SIZE;
            else if(j >= i % WORD_SIZE)
                sz = i % WORD_SIZE;
            else
                sz = j;
            
            i -= sz;
            j -= sz;
            vol_memcpy((char *)plog->start + i, (char *)var + j, sz);
        }
        
        // Calculate size of stack after removal
        int new_head = head / BLOCK_SIZE * WORD_SIZE + head % WORD_SIZE - size;
        // Move head to appropriate position
        int offset = (tid < WARP_SIZE ? 0 : plog->plog->tail[tid - WARP_SIZE]) + (tid % WARP_SIZE) * WORD_SIZE;
        new_head = new_head / WORD_SIZE * BLOCK_SIZE + new_head % WORD_SIZE + offset;
        // Drain if necessary
        if(PMEMLOG_UNSTRICT & plog->plog->flags) {   
            gpm_memcpy_nodrain(&plog->plog->head[tid], &new_head, sizeof(size_t), cudaMemcpyDeviceToDevice);
        }
        else {
            gpm_memcpy(&plog->plog->head[tid], &new_head, sizeof(size_t), cudaMemcpyDeviceToDevice);
        }
        
        return size;
    }
}

static __device__ void gpmlog_clear(gpmlog *plog, int partition = -1)
{
    if(plog->plog->flags & PMEMLOG_UNMANAGED) {
        if(partition == -1) {
            return;
        }
        gpmlog_clear_manual(plog, partition);
    }
    else {
        gpmlog_clear_managed(plog, partition);
    }
}


static __device__ int gpmlog_is_empty(gpmlog *plog, int partition = -1)
{
    bool empty = false;
    if(plog->plog->flags & PMEMLOG_UNMANAGED) {
        if(partition == -1)
            return -1;
        bool done = false;
        do {
            // Lock partition
            int locked = atomicCAS(&plog->locks[partition], 0, 1);
            __threadfence();
            if(!locked) {
                done = true;
                
                size_t start = plog->plog->head[partition];
                if((partition == 0 && start <= 0) || (partition != 0 && start <= plog->plog->tail[partition - 1]))
                    empty = true;
                
                __threadfence();
                atomicExch(&plog->locks[partition], 0);
            }
        } while(!done);
    }
    else {
        int tid = getGlobalIdx();
        int head = plog->plog->head[tid] - (tid < WARP_SIZE ? 0 : plog->plog->tail[tid - WARP_SIZE]);
        if(head / BLOCK_SIZE * WORD_SIZE + head % WORD_SIZE <= 0) {
            empty = true;
        }
    }
    return empty;
}


static __device__ size_t gpmlog_get_size(gpmlog *plog, int partition = -1)
{
    size_t size = 0;
    if(plog->plog->flags & PMEMLOG_UNMANAGED) {
        if(partition == -1) {
            return 0;
        }
        bool done = false;
        do {
            // Lock partition
            int locked = atomicCAS(&plog->locks[partition], 0, 1);
            __threadfence();
            if(!locked) {
                done = true;
                
                size = plog->plog->head[partition];
                
                if(partition != 0)
                    size -= plog->plog->tail[partition - 1];
                
                __threadfence();
                atomicExch(&plog->locks[partition], 0);
            }
        } while(!done);
    }
    else {
        int tid;
        if(partition == -1)
            tid = getGlobalIdx();
        else
            tid = partition;
        size_t head = plog->plog->head[tid] - (tid < WARP_SIZE ? 0 : plog->plog->tail[tid - WARP_SIZE]);
        //printf("Head at %lu\n", head);
        size = head / BLOCK_SIZE * WORD_SIZE + head % WORD_SIZE;
    }
    return size;
}

static __host__ __device__ int gpmlog_get_partitions(gpmlog *plog)
{
#if defined(__CUDA_ARCH__)
    return plog->plog->partitions;
#else
    gpmlog dummy;
    cudaMemcpy(&dummy, plog, sizeof(gpmlog), cudaMemcpyDeviceToHost);
    gpu_gpmlog nv_log;
    cudaMemcpy(&nv_log, dummy.plog, sizeof(gpu_gpmlog), cudaMemcpyDeviceToHost);
    return nv_log.partitions;
#endif
}
