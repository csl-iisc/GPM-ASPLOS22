#ifndef PMEM_HELPER
#define PMEM_HELPER
#include "libpmem.h"
#include <fstream>
#include <stdio.h>
#include <thread>

#define checkForCudaErrors(ans) { gpuAsserter((ans), __FILE__, __LINE__); }
 inline void gpuAsserter(cudaError_t code, const char *file, int line, bool abort=true)
 {
    if (code != cudaSuccess)
    {
       fprintf(stderr,"GPUassert %d: %s %s %d\n", code, cudaGetErrorString(code), file, line);
       if (abort) exit(code);
    } 
 }

enum PMEM_FLAGS {
    PMEM_NO_FUNCTIONAL = 1,
};

static cudaError_t create_gpm_file(const char *path, void **var, size_t size)
{
    int is_pmem;
    // Create a pmem file and memory map it
    if ((*var = pmem_map_file(path, size,
        PMEM_FILE_CREATE, 0666, &size, &is_pmem)) == NULL) {
        perror("pmem_map_file");
        exit(1);
    }
#ifdef DEBUG
    printf("Created pmem (%d) file at %s of size %ld\n", is_pmem, path, size);
#endif
    // Map to GPU address space
    return cudaHostRegister(*var, size, 0);
}

static cudaError_t open_gpm_file(const char *path, void **var, size_t &size)
{
    int is_pmem;
    // Open a pmem file and memory map it
    if ((*var = pmem_map_file(path, size,
        0, 0666, &size, &is_pmem)) == NULL) {
        perror("pmem_map_file");
        exit(1);
    }
#ifdef DEBUG
    printf("Opened pmem (%d) file at %s of size %ld\n", is_pmem, path, size);
#endif
    // Map to GPU address space
    return cudaHostRegister(*var, size, 0);
}

static cudaError_t close_gpm_file(void *var, size_t size)
{
    cudaError_t err = cudaHostUnregister(var);
    pmem_persist(var, size);
    if(pmem_unmap(var, size)) {
        printf("Error unmapping during file close (%s)\n", pmem_errormsg());
        exit(1);
    }
    // Map to GPU address space
    return err;
}
#endif
