#include "libgpmlog.cuh"
#include "libgpm.cuh"
#include <stdio.h>
#include <time.h>
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
__global__ void setup_log(gpmlog *plog, char *start, int len)
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

__device__ int getGlobalIdx()
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
        
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x) + threadIdx.x;
        
    return threadId;
}

__device__ int gpmlog_insert_manual(gpmlog *plog, void *var, size_t size, int partition)
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

__device__ int gpmlog_insert_managed(gpmlog *plog, void *var, size_t size, int partition)
{    
    int tid;
    if(partition == -1)
        tid = getGlobalIdx();
    else
        tid = partition;
    PMEM_READ_OP( size_t temp_head = plog->plog->head[tid] , sizeof(size_t) )
    
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

__device__ int gpmlog_read_manual(gpmlog *plog, void *var, size_t size, int partition)
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

__device__ int gpmlog_read_managed(gpmlog *plog, void *var, size_t size)
{    
    int tid = getGlobalIdx();
    
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
    return size;
}

__device__ int gpmlog_remove_manual(gpmlog *plog, size_t size, int partition)
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

__device__ int gpmlog_remove_managed(gpmlog *plog, size_t size)
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

__device__ void gpmlog_clear_manual(gpmlog *plog, int partition)
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

__device__ void gpmlog_clear_managed(gpmlog *plog)
{
    int tid = getGlobalIdx();
    PMEM_READ_OP( , sizeof(size_t) )
    size_t head = (tid % WARP_SIZE) * WORD_SIZE + (tid < WARP_SIZE ? 0 : plog->plog->tail[tid - WARP_SIZE]);
    
    gpm_memcpy_nodrain(&plog->plog->head[tid], &head, sizeof(size_t), cudaMemcpyDeviceToDevice);
    
    // Drain if necessary
    if(!(PMEMLOG_UNSTRICT & plog->plog->flags))
        gpm_drain();
}

/*******************
 *
 *  USER FUNCTIONS
 *
 *******************/


__host__ gpmlog *gpmlog_create(const char *path, size_t len, int partitions, int flags)
{
    // Mark as unmanaged
    flags |= PMEMLOG_UNMANAGED;
    
    // Create volatile metadata
    gpmlog *plog;
    // TODO: make this cudaMalloc
    cudaMallocManaged((void **)&plog, sizeof(gpmlog));
    plog->path = path;
    
    // Calculate log size = size of metadata + (heads + tails) + actual log
    plog->log_size = sizeof(gpu_gpmlog) + (2 * sizeof(size_t) * partitions) + len;
    
    // Create persistent memory chunk for persistent part
    void *log_pointer = gpm_map_file(path, plog->log_size, 1);
    int *locks;
    cudaMalloc((void **)&locks, sizeof(int) * partitions);
    
    // Set appropriate pointer locations
    plog->plog = (gpu_gpmlog *)log_pointer;
    plog->locks = locks;
    
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
    plog->start = log_pointer;
    
    cudaMemcpy(plog->plog, &temp_log, sizeof(gpu_gpmlog), cudaMemcpyHostToDevice);
    
    // Could parallelize this part
    // Now split the log among the partitions
    size_t *head = new size_t[partitions];
    size_t *tail = new size_t[partitions];
    head[0] = 0;
    for(int i = 1; i < partitions; ++i) {
        // Head = previous head + size of partition
        tail[i - 1] = head[i] = head[i - 1] + len / partitions;
    }
    // Last partition may have less/more if it doesnt divide evenly
    tail[partitions - 1] = len;
    
    // Copy values to persistent memory region
    cudaMemcpy(temp_log.head, head, sizeof(size_t) * partitions, cudaMemcpyHostToDevice);
    cudaMemcpy(temp_log.tail, tail, sizeof(size_t) * partitions, cudaMemcpyHostToDevice);
    
    return plog;
}

__host__ gpmlog *gpmlog_create_managed(const char *path, size_t &len, int blocks, int threads, int flags)
{
    gpmlog *plog;
    // TODO: make this cudaMalloc
    cudaMallocManaged((void **)&plog, sizeof(gpmlog));
    plog->path = path;
    
    // Convert len into BLOCK_SIZE byte blocks
    len = (len + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    int num_warps = (blocks * ((threads + WARP_SIZE - 1) / WARP_SIZE));
    if(len < BLOCK_SIZE * num_warps)
        len = BLOCK_SIZE * num_warps;
    
    int header_size = sizeof(gpu_gpmlog) + 2 * sizeof(size_t) * blocks * threads;
    // Calculate log size = size of metadata + (heads + tails) + offset + actual log (starting at BLOCK_SIZE byte offset)
    plog->log_size = header_size + (header_size % BLOCK_SIZE != 0 ? BLOCK_SIZE - header_size % BLOCK_SIZE : 0) + len;
    
    // Create persistent memory chunk for persistent part
    void *log_pointer = gpm_map_file(path, plog->log_size, 1);
    
    // Set appropriate pointer locations
    plog->plog = (gpu_gpmlog *)log_pointer;
    
    // Shift appropriate amount for next pointer
    log_pointer = (char *)log_pointer + sizeof(gpu_gpmlog);
    
    // Assign values for persistent log metadata
    gpu_gpmlog temp_log;
    temp_log.partitions = num_warps;
    temp_log.head       = (size_t *)log_pointer;
    log_pointer = (char *)log_pointer + sizeof(size_t) * blocks * threads;
    temp_log.tail       = (size_t *)log_pointer;
    temp_log.byte_size  = len;
    temp_log.flags      = flags;
    
    log_pointer = (char *)log_pointer + sizeof(size_t) * blocks * threads + 
        (header_size % BLOCK_SIZE != 0 ? BLOCK_SIZE - header_size % BLOCK_SIZE : 0);
    plog->start = log_pointer;
    
    cudaMemcpy(plog->plog, &temp_log, sizeof(gpu_gpmlog), cudaMemcpyHostToDevice);
    
    // Now split the log among the partitions
    size_t *head = new size_t[blocks * threads];
    size_t *tail = new size_t[blocks * threads];
    
    // Could parallelize this part
    // Divide BLOCK_SIZE byte regions among warps
    int regions_per_warp = len / (BLOCK_SIZE * num_warps);
    clock_t start = clock();
    head[0] = 0;
    tail[0] = regions_per_warp * BLOCK_SIZE;
    for(int i = 0; i < blocks; ++i) {
        for(int j = 0; j < threads; ++j) {
            if(i == 0 && j == 0)
                continue;
            if(j % WARP_SIZE != 0) {// Thread belongs to a previous warp
                // Each thread in a warp shifts by a word
                head[i * threads + j] = head[i * threads + j - 1] + WORD_SIZE;
                // Tails in a warp match
                tail[i * threads + j] = tail[i * threads + j - 1];
            }
            else {// New warp to be initialize
                // Start off where the previous warp left off
                head[i * threads + j] = tail[i * threads + j - 1];
                tail[i * threads + j] = head[i * threads + j] + regions_per_warp * BLOCK_SIZE;
            }
        }
    }
    
    // Set tail to end for last warp
    for(int j = threads - ((threads % WARP_SIZE == 0) ? WARP_SIZE : threads % WARP_SIZE); j < threads; ++j)
        tail[(blocks - 1) * threads + j] = len;
    
    // Copy values to persistent memory region
    cudaMemcpy(temp_log.head, head, sizeof(size_t) * blocks * threads, cudaMemcpyHostToDevice);
    cudaMemcpy(temp_log.tail, tail, sizeof(size_t) * blocks * threads, cudaMemcpyHostToDevice);
    
    return plog;
}

__host__ gpmlog *gpmlog_open(const char *path)
{
    size_t len = 0;
    char *start = (char *)gpm_map_file(path, len, false);
    // Create volatile metadata
    gpmlog *plog;
    // TODO: make this cudaMalloc
    cudaMallocManaged((void **)&plog, sizeof(gpmlog));
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
    return plog;
}

__host__ void gpmlog_close(gpmlog *plog)
{
    gpm_unmap(plog->path, plog->plog, plog->log_size);
    cudaFree(plog);
}

__device__ int gpmlog_insert(gpmlog *plog, void *var, size_t size, int partition)
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

__device__ int gpmlog_read(gpmlog *plog, void *var, size_t size, int partition)
{
    if(plog->plog->flags & PMEMLOG_UNMANAGED) {
        if(partition == -1) {
            return -1;
        }
        return gpmlog_read_manual(plog, var, size, partition);
    }
    else {
        return gpmlog_read_managed(plog, var, size);
    }
}

__device__ int gpmlog_remove(gpmlog *plog, size_t size, int partition)
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


__device__ int gpmlog_read_remove(gpmlog *plog, void *var, size_t size, int partition)
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

__device__ void gpmlog_clear(gpmlog *plog, int partition)
{
    if(plog->plog->flags & PMEMLOG_UNMANAGED) {
        if(partition == -1) {
            return;
        }
        gpmlog_clear_manual(plog, partition);
    }
    else {
        gpmlog_clear_managed(plog);
    }
}


__device__ int gpmlog_is_empty(gpmlog *plog, int partition)
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


__device__ size_t gpmlog_get_size(gpmlog *plog, int partition)
{
    size_t size = 0;
    if(plog->plog->flags & PMEMLOG_UNMANAGED) {
        if(partition == -1) {
            return -1;
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
        int tid = getGlobalIdx();
        int head = plog->plog->head[tid] - (tid < WARP_SIZE ? 0 : plog->plog->tail[tid - WARP_SIZE]);
        size = head / BLOCK_SIZE * WORD_SIZE + head % WORD_SIZE;
    }
    return size;
}
