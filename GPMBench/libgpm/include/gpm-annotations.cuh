#pragma once
#include <stdint.h>

#define SPLITS (72.0 * 1024.0)
#define FREQ (2.1)
#define CALC(base_bw, target_bw, size) (SPLITS * FREQ * size * (1.0 / target_bw - 1.0 / base_bw))

#if defined(NVM_ALLOC_GPU) && defined(EMULATE_NVM) && defined(EMULATE_NVM_BW)
    #define BW_DELAY(cycles) \
    /*{\
        size_t start = clock64();\
        while(clock64() - start < cycles);\
    }
    */
#else
    #define BW_DELAY(cycles)
#endif

//#define OUTPUT_NVM_DETAILS
#if (defined(NVM_ALLOC_GPU) || defined(FAKE_NVM)) && defined(EMULATE_NVM)
    #define GPU_FREQ (2.1)
    #define NUM_ENTRIES (32 * 256)
    #define DELAY (long long int)(200 * GPU_FREQ)

    static __device__ uint64_t nvm_write[NUM_ENTRIES];
    #ifdef NVM_ALLOC_GPU
        #define PMEM_WRITE_OP(operation, size) \
            operation;
            /*atomicAdd_block((unsigned long long*)&nvm_write[(threadIdx.x + blockDim.x * blockIdx.x) / 32 % NUM_ENTRIES], size);\*/
    #else
        #define PMEM_WRITE_OP(operation, size) \
            operation;
    #endif

    #define PMEM_READ_OP(operation, size) \
        {\
            long long int start = clock64();\
            while(clock64() - start < DELAY * (size) / 4);\
        }\
        operation;
    #ifdef NVM_ALLOC_GPU
        #define PMEM_READ_WRITE_OP(operation, size) \
        {\
            long long int start = clock64();\
            while(clock64() - start < DELAY * (size) / 4);\
        }\
        //atomicAdd_block((unsigned long long*)&nvm_write[(threadIdx.x + blockDim.x * blockIdx.x) / 32 % NUM_ENTRIES], size);\
        /*Do stuff*/    \
        operation;
    #else
        #define PMEM_READ_WRITE_OP(operation, size) \
        {\
            long long int start = clock64();\
            while(clock64() - start < DELAY * (size) / 4);\
        }\
        /*Do stuff*/   \
        operation;
    #endif
        
    #define OUTPUT_STATS

#elif OUTPUT_NVM_DETAILS

	#ifndef GPUDB
		static __device__ uint64_t nvm_write;
		static __device__ uint64_t nvm_read;
	#else
		extern __device__ uint64_t nvm_write;
		extern __device__ uint64_t nvm_read;
	#endif
#define PMEM_WRITE_OP(operation, size) \
    operation;\
    atomicAdd((unsigned long long*)&nvm_write, size);\
    /*Do stuff*/
    
#define PMEM_READ_OP(operation, size) \
    atomicAdd((unsigned long long*)&nvm_read, size);\
    /*Do stuff*/    \
    operation;
    
#define PMEM_READ_WRITE_OP(operation, size) \
    atomicAdd((unsigned long long*)&nvm_read, size);\
    operation;\
    atomicAdd((unsigned long long*)&nvm_write, size);\
    /*Do stuff*/    
    
#define OUTPUT_STATS \
    {\
        uint64_t tot_write;\
        uint64_t tot_read;\
        cudaMemcpyFromSymbol(&tot_write, nvm_write, sizeof(uint64_t));\
        cudaMemcpyFromSymbol(&tot_read, nvm_read, sizeof(uint64_t));\
        uint64_t j = 0;\
        cudaMemcpyToSymbol(nvm_write, &j, sizeof(uint64_t));\
        cudaMemcpyToSymbol(nvm_read, &j, sizeof(uint64_t));\
        printf("Tot writes\t%ld\ttot reads\t%ld\n", tot_write, tot_read);\
    }
    
#else

#define PMEM_WRITE_OP(operation, size) \
    /*Do stuff*/    \
    operation;
    
#define PMEM_READ_OP(operation, size) \
    /*Do stuff*/    \
    operation;
    
#define PMEM_READ_WRITE_OP(operation, size) \
    /*Do stuff*/    \
    operation;
    
#define OUTPUT_STATS
#endif
