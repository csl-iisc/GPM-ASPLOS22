#include "gpm-helper.cuh"
#include "libpmem.h"
#include <fstream>
#include <stdio.h>
#include <thread>

/*
 * Stuff used for emulation of NVM
 *
 * Variables:
 * - PMEM_START_HOST: Starting address of emulated block of NVM in host mem
 * - PMEM_START_DEV: Starting address of emulated block of NVM in dev mem (same value as PMEM_START_HOST)
 * - gpm_start_alloc: Offset up to which NVM has already been allocated
 * - gpm_init_complete: Whether emulated NVM block has been allocated
 *
 * Functions:
 * - init_gpm: Allocate chunk of memory to be used as emulated NVM
 * - get_memory: Return address to chunk of emulated NVM of requested size (256B aligned)
 * - free_memory: Free memory chunk passed.
 *
 * For simplicity, memory allocation/deallocation only works with contiguous chunks.
 *
 */

#if defined(NVM_ALLOC_GPU) || defined(FAKE_NVM)
void *PMEM_START_HOST; 
__constant__ char *PMEM_START_DEV = 0;
char *gpm_start_alloc; // Point from which memory allocation can begin
bool gpm_init_complete = false;

// Allocate block of memory designated as persistent memory
cudaError_t init_gpm(int flags)
{
    gpm_init_complete = true;
#ifndef NVM_ALLOC_GPU
    cudaError_t err = cudaMallocHost((void **)&PMEM_START_HOST, PMEM_SIZE);
#else
    size_t dummy;
    cudaError_t err = cudaMallocPitch((void **)&PMEM_START_HOST, &dummy, PMEM_SIZE, 1);
#endif
    if(err != cudaSuccess)
        return err;
    gpm_start_alloc = (char *)PMEM_START_HOST;
    // Copy pointer starting point to device
    err = cudaMemcpyToSymbol(PMEM_START_DEV, &PMEM_START_HOST, sizeof(char *));
    cudaDeviceSynchronize();
    return err;
}

// (internal) Allocate memory from persistent partition
// Currently only supports contiguous allocation, without
// supporting free. Should be modified later to track free
// and allocated memory in possible tree-like structure
cudaError_t get_memory(void **var, size_t size)
{
    if(gpm_start_alloc + (size + 255) / 256 * 256 <= (char *)PMEM_START_HOST + PMEM_SIZE) {
        *var = gpm_start_alloc;
        gpm_start_alloc += (size + 255) / 256 * 256;
        return cudaSuccess;
    }
    else {
        return cudaErrorMemoryAllocation;
    }
}

// (internal) Deallocate memory from persistent partition
cudaError_t free_memory(void *var, size_t size)
{
    if(Is_pmem(var, size) && (char *)var + (size + 255) / 256 * 256 == gpm_start_alloc) {
        gpm_start_alloc = (char *)var;
        return cudaSuccess;
    }
    return cudaErrorInvalidValue;
}
#endif

cudaError_t create_gpm_file(const char *path, void **var, size_t size)
{
#if defined(NVM_ALLOC_GPU) || defined(FAKE_NVM)
    if(!gpm_init_complete) {
        cudaError_t err = init_gpm();
        if(err != cudaSuccess)
            return err;
    }
    
    // Allocate required memory for file
    return get_memory(var, size);
#else
    char *full_path = new char[sizeof("/pmem/") + strlen(path)];
    strcpy(full_path, "/pmem/");
    strcat(full_path, path);
    int is_pmem;
    // Create a pmem file and memory map it
    if ((*var = pmem_map_file(full_path, size,
        PMEM_FILE_CREATE, 0666, &size, &is_pmem)) == NULL) {
        perror("pmem_map_file");
        exit(1);
    }
    printf("Created pmem (%d) file at %s of size %ld\n", is_pmem, path, size);
    // Map to GPU address space
    return cudaHostRegister(*var, size, 0);
#endif
}

cudaError_t open_gpm_file(const char *path, void **var, size_t &size)
{
#if defined(NVM_ALLOC_GPU) || defined(FAKE_NVM)
    if(!gpm_init_complete) {
        cudaError_t err = init_gpm();
        if(err != cudaSuccess)
            return err;
    }
    
    // Read from persistent file
    std::ifstream file(path, std::ifstream::binary);
    // Calculate size of file
    file.ignore( std::numeric_limits<unsigned int>::max() );
    size = file.gcount();
    file.clear();   //  Since ignore will have set eof.
    file.seekg( 0, std::ios_base::beg );
    
    char *input = new char[size];
    file.read(input, size);
    
    // Allocate required memory for file
    cudaError_t err = get_memory(var, size);
    
    if(err != cudaSuccess)
        return err;
    
    return cudaMemcpy(*var, input, size, cudaMemcpyHostToDevice);
#else
    char *full_path = new char[sizeof("/pmem/") + strlen(path)];
    strcpy(full_path, "/pmem/");
    strcat(full_path, path);
    int is_pmem;
    // Open a pmem file and memory map it
    if ((*var = pmem_map_file(full_path, size,
        0, 0666, &size, &is_pmem)) == NULL) {
        perror("pmem_map_file");
        exit(1);
    }
    printf("Opened pmem (%d) file at %s of size %ld\n", is_pmem, path, size);
    // Map to GPU address space
    return cudaHostRegister(*var, size, 0);
#endif
}

cudaError_t close_gpm_file(const char *path, void *var, size_t size)
{
#if defined(NVM_ALLOC_GPU) || defined(FAKE_NVM)
    /*std::ofstream file(path, std::ofstream::binary);
    char *output = new char[size];
    
    cudaError_t err = cudaMemcpy(output, var, size, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
        return err;
    file.write(output, size);*/
    return free_memory(var, size);
#else
    cudaError_t err = cudaHostUnregister(var);
    pmem_persist(var, size);
    if(pmem_unmap(var, size)) {
        printf("Error unmapping during file close (%s)\n", pmem_errormsg());
        exit(1);
    }
    // Map to GPU address space
    return err;
#endif
}

// Internal function to determine if region is persistent
__device__ __host__ bool Is_pmem(const void *addr, size_t len)
{
#if defined(NVM_ALLOC_GPU) || defined(FAKE_NVM)
    #if defined(__CUDA_ARCH__)
        // Device code here
        return addr >= (char *)PMEM_START_DEV && (char *)addr + len < (char *)PMEM_START_DEV + PMEM_SIZE;
    #else
        // Host code here
        return addr >= PMEM_START_HOST && (char *)addr + len <= (char *)gpm_start_alloc;
    #endif
#else
    #if defined(__CUDA_ARCH__)
        // No way to reliably check if memory is pmem from inside GPU kernel for GPM-far
        return true;
    #else
        // Host code here
        return pmem_is_pmem(addr, len);
    #endif
#endif
}
