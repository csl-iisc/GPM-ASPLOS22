#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

__host__ void *gpm_map_file(const char *path, size_t *len, int create);
__host__ cudaError_t gpm_unmap(const char *path, void *addr, size_t len);
__device__ void gpm_drain(void);
__device__ void gpm_flush(const void *addr, size_t len);
__device__ void gpm_persist(const void *addr, size_t len);
__device__ __host__ int gpm_is_gpm(const void *addr, size_t len);
__device__ __host__ cudaError_t gpm_memcpy(void *gpmdest, const void *src, size_t len, enum cudaMemcpyKind kind);
__device__ __host__ cudaError_t gpm_memset(void *gpmdest, unsigned char  c, size_t len);
__device__ __host__ cudaError_t gpm_memcpy_nodrain(void *gpmdest, const void *src, size_t len, enum cudaMemcpyKind kind);
__device__ __host__ cudaError_t gpm_memset_nodrain(void *gpmdest, unsigned char c, size_t len);

// Helper function to avoid CUDA's inefficient memcpy
__device__ void vol_memcpy(void *dest, const void *src, size_t len);
