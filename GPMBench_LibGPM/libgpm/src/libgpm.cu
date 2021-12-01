#include "gpm-helper.cuh"
#include "libgpm.cuh"
#include <stdio.h>
#include <assert.h>

#define DWORD unsigned long long
#define WORD unsigned int
#define BYTE unsigned char

/*
 * gpm_map_file -- create persistent file-mapped memory
 */
__host__ void *gpm_map_file(const char *path, size_t &len, bool create)
{
    if (create && len <= 0) {
        printf("invalid file length %zu", len);
        return NULL;
    }

    if (len != 0 && !create) {
        printf("non-zero 'len' not allowed without create option");
        return NULL;
    }

    void *addr;
    cudaError_t err;
    
    if(create)
        err = create_gpm_file(path, &addr, len);
    else
        err = open_gpm_file(path, &addr, len);
    
    if(err != cudaSuccess)
        printf("CUDA Error %d while trying to create gpm file", err);
    
    return addr;
}

/*
 * gpm_unmap -- close and persist the file-mapped memory
 */
__host__ cudaError_t gpm_unmap(const char *path, void *addr, size_t len)
{
    assert(Is_pmem(addr, len));
    return close_gpm_file(path, addr, len);
}

/*
 * gpm_drain -- wait for any PM stores to drain from HW buffers
 */
__device__ void gpm_drain(void)
{
#ifdef NVM_ALLOC_GPU
    __threadfence();
    
    #ifdef EMULATE_NVM
        unsigned long long num = atomicExch_block((unsigned long long*)&nvm_write[(threadIdx.x + blockDim.x * blockIdx.x) / 32 % NUM_ENTRIES], 0);
        // 150 cycle delay - 35 cycle atomic
        num = (num + 31) / 32 * 115;
        unsigned long long start = clock64();
        while(clock64() - start < num);
    #endif
#else
    // Drain caches
    __threadfence_system();
#endif
    return;
}

/*
 * gpm_flush -- flush cache for the given range
 */
__device__ void gpm_flush(const void *addr, size_t len)
{
    assert(Is_pmem(addr, len));
    
    int i = 0;
    
    volatile DWORD *b = (DWORD *)((BYTE *)addr + i);
    // Store elements at word granularity
    for(; i + sizeof(DWORD) <= len && 
        ((size_t)addr % sizeof(DWORD)) == 0
        ; i += sizeof(DWORD), b += 1) {
        // Store the value in addr to a volatile copy 
        // of itself to guarantee cache flush
        PMEM_WRITE_OP( (*b) = *((DWORD *)addr + (i / sizeof(DWORD))) , sizeof(DWORD) )
    }
    
    volatile WORD *c = (WORD *)((BYTE *)addr + i);    
    // Store elements at word granularity
    for(; i + sizeof(WORD) <= len && 
        ((size_t)addr % sizeof(WORD)) == 0
        ; i += sizeof(WORD), c += 1) {
        // Store the value in addr to a volatile copy 
        // of itself to guarantee cache flush
        PMEM_WRITE_OP( (*c) = *((WORD *)addr + (i / sizeof(WORD))) , sizeof(WORD) )
    }
    
    volatile BYTE *d = ((BYTE *)addr + i);
    // Store remaining elements at byte granularity
    for(; i < len; i += sizeof(BYTE), d += 1) {
        // Store the value in addr to a volatile copy 
        // of itself to guarantee cache flush
       PMEM_WRITE_OP( (*d) = *((BYTE *)addr + i) , sizeof(BYTE) )
    }
}

/*
 * gpm_persist -- make any cached changes to a range of gpm persistent
 */
__device__ void gpm_persist(const void *addr, size_t len)
{
    gpm_flush(addr, len);
    gpm_drain();
}

/*
 * gpm_is_gpm -- return true if entire range is persistent memory
 */
__device__ __host__ int gpm_is_pmem(const void *addr, size_t len)
{
    return Is_pmem(addr, len);
}

/*
 * gpm_memcpy_nodrain --  memcpy to gpm without hw drain
 */
__device__ cudaError_t gpm_memcpy_nodrain(void *gpmdest, const void *src, size_t len, cudaMemcpyKind kind)
{
#if defined(__CUDA_ARCH__)
    // Device code here
    assert(Is_pmem(gpmdest, len));
    
    int i = 0;
    
    volatile DWORD *b = (DWORD *)((BYTE *)gpmdest + i);
    // Store elements at word granularity
    for(; i + sizeof(DWORD) <= len && 
        ((size_t)gpmdest % sizeof(DWORD)) == 0 && 
        ((size_t)src % sizeof(DWORD)) == 0
        ; i += sizeof(DWORD), b += 1) {
        // Store the value in addr to a volatile copy 
        // of itself to guarantee cache flush
        PMEM_WRITE_OP( (*b) = *((DWORD *)src + (i / sizeof(DWORD))) , sizeof(DWORD) )
    }
    
    volatile WORD *c = (WORD *)((BYTE *)gpmdest + i);
    // Copy elements at word granularity
    for(; i + sizeof(WORD) <= len && 
        ((size_t)gpmdest % sizeof(WORD)) == 0 && 
        ((size_t)src % sizeof(WORD)) == 0
        ; i += sizeof(WORD), c += 1) {
        // Store the value to a volatile copy to guarantee cache flush
        PMEM_WRITE_OP( (*c) = *((WORD *)src + (i / sizeof(WORD))) , sizeof(WORD) )
    }
    
    volatile BYTE *d = ((BYTE *)gpmdest + i);
    // Store remaining elements at byte granularity
    for(; i < len; i += sizeof(BYTE), d += 1) {
        // Store the value to a volatile copy to guarantee cache flush
        PMEM_WRITE_OP( (*d) = *((BYTE *)src + i) , sizeof(BYTE) )
    }
    return cudaSuccess;
#else
    // Host code here
    assert(Is_pmem(gpmdest, len));
    return cudaMemcpy(gpmdest, src, len, kind);
#endif
}

/*
 * gpm_memset_nodrain -- memset to gpm without hw drain
 */
__device__ __host__ cudaError_t gpm_memset_nodrain(void *gpmdest, unsigned char value, size_t len)
{
#if defined(__CUDA_ARCH__)
    // Device code here
    assert(Is_pmem(gpmdest, len));
    
    int i = 0;
    
    // Can only set at word granularity for 0, as memset is supposed to copy byte-by-byte
    if(value == 0)
    {
        volatile DWORD *b = (DWORD *)((BYTE *)gpmdest + i);
        // Store elements at word granularity
        for(; i + sizeof(DWORD) <= len && 
        ((size_t)gpmdest % sizeof(DWORD)) == 0
        ; i += sizeof(DWORD), b += 1) {
            // Store the value in addr to a volatile copy 
            // of itself to guarantee cache flush
            PMEM_WRITE_OP( (*b) = value , sizeof(DWORD) )
        }

        volatile WORD *c = (WORD *)((BYTE *)gpmdest + i);
        // Copy elements at word granularity
        for(; i + sizeof(WORD) <= len && 
        ((size_t)gpmdest % sizeof(WORD)) == 0
        ; i += sizeof(WORD), c += 1) {
            // Store the value to a volatile copy to guarantee cache flush
            PMEM_WRITE_OP( (*c) = value , sizeof(WORD) )
        }
    }
    
    volatile BYTE *d = ((BYTE *)gpmdest + i);
    // Store remaining elements at byte granularity
    for(; i < len; i += sizeof(BYTE), d += 1) {
        // Store the value to a volatile copy to guarantee cache flush
        PMEM_WRITE_OP( (*d) = value , sizeof(BYTE) )
    }
    return cudaSuccess;
#else
    // Host code here
    assert(Is_pmem(gpmdest, len));
    return cudaMemset(gpmdest, value, len);
#endif
}

__global__ void gpm_memcpyKernel(void *gpmdest, const void *src, size_t len, cudaMemcpyKind kind)
{
    int TID = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = TID * 4; i < len; i += blockDim.x * gridDim.x * 4) {
        gpm_memcpy_nodrain((char *)gpmdest + i, (char *)src + i, min((size_t)4, len - i), kind);
    }
    gpm_drain();
}

/*
 * gpm_memcpy --  memcpy to gpm
 */
__device__ __host__ cudaError_t gpm_memcpy(void *gpmdest, const void *src, size_t len, cudaMemcpyKind kind)
{
#if defined(__CUDA_ARCH__)
    // Device code here
    gpm_memcpy_nodrain(gpmdest, src, len, kind);
    gpm_drain();
    return cudaSuccess;
#else
    // Host code here
    assert(Is_pmem(gpmdest, len));
    // If data is a host variable, move to device first
    if(kind == cudaMemcpyHostToDevice || cudaMemcpyHostToHost) {
        void *d_src;
        cudaError_t err = cudaMalloc((void **)&d_src, len);
        if(err != cudaSuccess) {
            printf("Error %d cudaMalloc in gpm_memcpy\n", err);
            return err;
        }
        err = cudaMemcpy(d_src, src, len, kind);
        if(err != cudaSuccess) {
            printf("Error %d cudaMemcpy in gpm_memcpy\n", err);
            return err;
        }
        gpm_memcpyKernel<<<((len + 3) / 4 + 1023) / 1024, 1024>>> (gpmdest, d_src, len, kind);
        cudaFree(d_src);
    }
    // If data is already device variable, just copy directly
    else if(kind == cudaMemcpyDeviceToDevice || kind == cudaMemcpyDeviceToHost) {
        gpm_memcpyKernel<<<((len + 3) / 4 + 1023) / 1024, 1024>>> (gpmdest, src, len, kind);
    }
    return cudaGetLastError();
#endif
}

__global__ void gpm_memsetKernel(void *gpmdest, unsigned char c, size_t len)
{
    int TID = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = TID * 4; i < len; i += blockDim.x * gridDim.x * 4)
        gpm_memset_nodrain((char *)gpmdest + i, c, min((size_t)4, len - i));
    gpm_drain();
}
/*
 * gpm_memset -- memset to gpm
 */
__device__ __host__ cudaError_t gpm_memset(void *gpmdest, unsigned char c, size_t len)
{
#if defined(__CUDA_ARCH__)
    // Device code here
    gpm_memset_nodrain(gpmdest, c, len);
    gpm_drain();
    return cudaSuccess;
#else
    // Host code here
    assert(Is_pmem(gpmdest, len));
    gpm_memsetKernel<<<((len + 3) / 4 + 1023) / 1024, 1024>>> (gpmdest, c, len);
    return cudaGetLastError();
#endif
}

// Helper function to avoid CUDA's inefficient memcpy
__device__ void vol_memcpy(void *dest, const void *src, size_t len)
{
    int i = 0;
    
    volatile DWORD *b = (DWORD *)((BYTE *)dest + i);
    // Store elements at double-word granularity if possible
    for(; i + sizeof(DWORD) <= len && 
        ((size_t)dest % sizeof(DWORD)) == 0 && 
        ((size_t)src % sizeof(DWORD) == 0)
        ; i += sizeof(DWORD), b += 1) {
        // Store the value in addr to a volatile copy 
        // of itself to guarantee cache flush
        (*b) = *((DWORD *)src + (i / sizeof(DWORD)));
    }
    
    volatile WORD *c = (WORD *)((BYTE *)dest + i);
    // Copy elements at word granularity if possible
    for(; i + sizeof(WORD) <= len && 
        ((size_t)dest % sizeof(WORD)) == 0 && 
        ((size_t)src % sizeof(WORD) == 0)
        ; i += sizeof(WORD), c += 1) {
        // Store the value to a volatile copy to guarantee cache flush
        (*c) = *((WORD *)src + (i / sizeof(WORD)));
    }
    
    volatile BYTE *d = ((BYTE *)dest + i);
    // Store remaining elements at byte granularity
    for(; i < len; i += sizeof(BYTE), d += 1) {
        // Store the value to a volatile copy to guarantee cache flush
        (*d) = *((BYTE *)src + i);
    }
}
