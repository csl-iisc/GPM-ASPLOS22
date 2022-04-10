# LibGPM
LibGPM contains the source of our CUDA library which provides a user-friendly interface for GPU-accelerated recoverable applications. 

## Libraries
### LibGPM (libgpm.cuh)
 - (__host__) void *gpm_map_file(const char *path, size_t &len, bool create)
	 - If create is true, creates memory-mapped file at path, with of len bytes size.
	 - If create is false, opens memory-mapped file at path, storing size of file in len. Input len should be 0.
	 - Returns pointer to persistent region in GPU of len size
 - (__host__) cudaError_t gpm_unmap(void *addr, size_t len)
	 - Unmaps memory-mapped variable with starting pointer addr and size len, persisting all data.
	 - Returns cudaSuccess on success or error code otherwise
 - (__host__) void gpm_persist_begin(void)
 	 - Turns DDIO off, allowing in-kernel persistence for further GPU kernels accessing PM.
 - (__host__) void gpm_persist_end(void)
 	 - Turns DDIO on, in-kernel persistence is no longer guaranteed.
 - (__device__) void gpm_persist()
 	 - Guarantees all prior writes to PM by the calling thread are persisted.
	 - Implemented using `__threadfence_system()`.
 - (__device__, __host__) cudaError_t gpm_memcpy_nodrain(void *gpmdest, const void *src, size_t len, cudaMemcpyKind kind)
	- Memcpys data from src to gpmdest of size len, without persisting the region. The variable kind indicates the type of memcpy (cudaMemcpyDeviceToDevice, cudaMemcpyHostToDevice, etc.).
 - (__device__, __host__) cudaError_t gpm_memset_nodrain(void *gpmdest, unsigned char value, size_t len)
	- Assigns value byte-by-byte to the region starting from pointer gpmdest of size len, without persisting the region.
 - (__device__, __host__) cudaError_t gpm_memcpy(void *pmemdest, const void *src, size_t len, cudaMemcpyKind kind)
	- Memcpys data from src to gpmdest of size len, guaranteeing persistance on completion. The variable kind indicates the type of memcpy (cudaMemcpyDeviceToDevice, cudaMemcpyHostToDevice, etc.).
	- Failure-atomicity is not guaranteed.
 - (__device__, __host__) cudaError_t pmem_memset(void *pmemdest, int c, size_t len)
	- Assigns value byte-by-byte to the region starting from pointer gpmdest of size len, guaranteeing persistance on completion.
	- Failure-atomicity is not guaranteed.
### LibGPMLog (libgpmlog.cuh)
 - TODO

### LibGPMCP (libgpmcp.cuh)
 - TODO

## Source code
This folder contains 6 files, which we explain below:
* [libgpm.cuh](libgpm.cuh) - Contains the main implementation details of GPM, which allow for allocation/deallocation on PMEM.
* [libgpmlog.cuh](libgpmlog.cuh) - Contains the relevant implementation of logging (HCL and conventional) in GPM.
* [libgpmcp.cuh](libgpmcp.cuh) - Contains the relevant implementation of checkpointing in GPM.
* [bandwidth_analysis.cuh](bandwidth_analysis.cuh) - Contains helper definitions used for bandwidth measurement.
* [change-ddio.h](change-ddio.h) - Contains functions to turn DDIO off and on. GPM wrappers for these functions are found in libgpm. [1]
* [gpm-helper.cuh](gpm-helper.cuh) - Contains common helper functions needed across other files.


References: 
[1] Characterizing and Optimizing Remote Persistent Memory with RDMA and NVM. Authors: Xingda Wei and Xiating Xie and Rong Chen and Haibo Chen and Binyu Zang. Published in: Usenix ATC'2021
