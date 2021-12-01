#ifndef PMEM_HELPER
#define PMEM_HELPER
// Size of persistent memory partition, default = 2 GiB
#define PMEM_SIZE ((size_t)2 * (size_t)1024 * (size_t)1024 * (size_t)1024)

enum PMEM_FLAGS {
    PMEM_NO_FUNCTIONAL = 1,
};

__constant__ extern void *PMEM_START_DEV;

// Create persistent memory partition in GPU
cudaError_t init_gpm(int flags = 0);

// Create and return file-backed memory in PM
cudaError_t create_gpm_file(const char *path, void **var, size_t size);
cudaError_t open_gpm_file(const char *path, void **var, size_t &size);
cudaError_t close_gpm_file(const char *path, void *var, size_t size);

// (internal) Allocate memory from persistent partition
cudaError_t get_memory(void **var, size_t size);

// (internal) Deallocate memory from persistent partition
cudaError_t free_memory(void *var, size_t size);

// Check whether memory is persistent memory
__device__ __host__ bool Is_gpm(const void *addr, size_t len);

#endif
