#pragma once
#include "gpm-helper.cuh"
#include "change-ddio.h"
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define DWORD unsigned long long
#define WORD unsigned int
#define BYTE unsigned char

/*
 * gpm_map_file -- create persistent file-mapped memory
 */
static __host__ void *gpm_map_file(const char *path, size_t &len, bool create)
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
static __host__ cudaError_t gpm_unmap(void *addr, size_t len)
{
    return close_gpm_file(addr, len);
}

/*
 * gpm_persist_begin -- start region where in-kernel PM accesses bypass DDIO
 */
static __host__ void gpm_persist_begin()
{
    ddio_off();
}

/*
 * gpm_persist_end -- end region where in-kernel PM accesses bypass DDIO
 */
static __host__ void gpm_persist_end()
{
	ddio_on();
}

/*
 * gpm_persist -- make any changes to a range of gpm persistent
 */
static __device__ void gpm_persist()
{
    // Drain caches
    __threadfence_system();
}

/*
 * gpm_memcpy_nodrain --  memcpy to gpm without hw drain
 */
static __device__ cudaError_t gpm_memcpy_nodrain(void *gpmdest, const void *src, size_t len, cudaMemcpyKind kind)
{
#if defined(__CUDA_ARCH__)
    // Device code here
    int i = 0;
    volatile DWORD *b = (DWORD *)((BYTE *)gpmdest + i);
    // Store elements at word granularity
    for(; i + sizeof(DWORD) <= len && 
        ((size_t)gpmdest % sizeof(DWORD)) == 0 && 
        ((size_t)src % sizeof(DWORD)) == 0
        ; i += sizeof(DWORD), b += 1) {
        // Store the value in addr to a volatile copy 
        // of itself to guarantee cache flush
        (*b) = *((DWORD *)src + (i / sizeof(DWORD)));
    }
    
    volatile WORD *c = (WORD *)((BYTE *)gpmdest + i);
    // Copy elements at word granularity
    for(; i + sizeof(WORD) <= len && 
        ((size_t)gpmdest % sizeof(WORD)) == 0 && 
        ((size_t)src % sizeof(WORD)) == 0
        ; i += sizeof(WORD), c += 1) {
        // Store the value to a volatile copy to guarantee cache flush
        (*c) = *((WORD *)src + (i / sizeof(WORD)));
    }
    
    volatile BYTE *d = ((BYTE *)gpmdest + i);
    // Store remaining elements at byte granularity
    for(; i < len; i += sizeof(BYTE), d += 1) {
        // Store the value to a volatile copy to guarantee cache flush
        (*d) = *((BYTE *)src + i);
    }
    return cudaSuccess;
#else
    // Host code here
    return cudaMemcpy(gpmdest, src, len, kind);
#endif
}

/*
 * gpm_memset_nodrain -- memset to gpm without hw drain
 */
static __device__ cudaError_t gpm_memset_nodrain(void *gpmdest, unsigned char value, size_t len)
{
#if defined(__CUDA_ARCH__)
    // Device code here    
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
            (*b) = value;
        }

        volatile WORD *c = (WORD *)((BYTE *)gpmdest + i);
        // Copy elements at word granularity
        for(; i + sizeof(WORD) <= len && 
        ((size_t)gpmdest % sizeof(WORD)) == 0
        ; i += sizeof(WORD), c += 1) {
            // Store the value to a volatile copy to guarantee cache flush
            (*c) = value;
        }
    }
    
    volatile BYTE *d = ((BYTE *)gpmdest + i);
    // Store remaining elements at byte granularity
    for(; i < len; i += sizeof(BYTE), d += 1) {
        // Store the value to a volatile copy to guarantee cache flush
        (*d) = value;
    }
    return cudaSuccess;
#else
    // Host code here
    return cudaMemset(gpmdest, value, len);
#endif
}

static __global__ void gpm_memcpyKernel(void *gpmdest, const void *src, size_t len, cudaMemcpyKind kind)
{
    int TID = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = TID * 4; i < len; i += blockDim.x * gridDim.x * 4) {
        gpm_memcpy_nodrain((char *)gpmdest + i, (char *)src + i, min((size_t)4, len - i), kind);
    }
    gpm_persist();
}

/*
 * gpm_memcpy --  memcpy to gpm
 */
static __device__ __host__ cudaError_t gpm_memcpy(void *gpmdest, const void *src, size_t len, cudaMemcpyKind kind)
{
#if defined(__CUDA_ARCH__)
    // Device code here
    gpm_memcpy_nodrain(gpmdest, src, len, kind);
    gpm_persist();
    return cudaSuccess;
#else
    // Host code here
    // If data is a host variable, move to device first
    if(kind == cudaMemcpyHostToDevice || cudaMemcpyHostToHost) {
        void *d_src;
        cudaError_t err = cudaMalloc((void **)&d_src, len);
        if(err != cudaSuccess) {
            printf("Error %d cudaMalloc in gpm_memcpy for %ld\n", err, len);
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

static __global__ void gpm_memsetKernel(void *gpmdest, unsigned char c, size_t len)
{
    int TID = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = TID * 4; i < len; i += blockDim.x * gridDim.x * 4)
        gpm_memset_nodrain((char *)gpmdest + i, c, min((size_t)4, len - i));
    gpm_persist();
}
/*
 * gpm_memset -- memset to gpm
 */
static __device__ __host__ cudaError_t gpm_memset(void *gpmdest, unsigned char c, size_t len)
{
#if defined(__CUDA_ARCH__)
    // Device code here
    gpm_memset_nodrain(gpmdest, c, len);
    gpm_persist();
    return cudaSuccess;
#else
    // Host code here
    gpm_memsetKernel<<<((len + 3) / 4 + 1023) / 1024, 1024>>> (gpmdest, c, len);
    return cudaGetLastError();
#endif
}

// Helper function to avoid CUDA's inefficient memcpy
static __device__ void vol_memcpy(void *dest, const void *src, size_t len)
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
