CUDA_DIR = /usr/local/cuda-11.0
 
# CUDA toolkit libraries
CUDA_LIB_DIR := $(CUDA_DIR)/lib
ifeq ($(shell uname -m), x86_64)
     ifeq ($(shell if test -d $(CUDA_DIR)/lib64; then echo T; else echo F; fi), T)
        CUDA_LIB_DIR := $(CUDA_DIR)/lib64
     endif
endif
 
SDK_DIR = $(CUDA_DIR)/samples
 
OPENCL_DIR = /if10/kw5na/Packages/AMD-APP-SDK-v2.8-RC-lnx64
OPENCL_INC = $(OPENCL_DIR)/include/
OPENCL_LIB = $(OPENCL_DIR)/lib/x86_64/ -lOpenCL

CC := nvcc


NVCC_FLAGS := -std=c++11 -lpthread -lnvidia-ml -lpmem -O3 -arch=sm_70 -lpci -Xcompiler -fopenmp

ifeq ($(dbg),1) 
NVCC_FLAGS += -g -G
endif

PMEM_LIB := ../../common/src/pmem-helper.cu ../../common/src/libpmem-g.cu ../../common/src/libpmemlog-g. cu ../../common/src/libpmemcp-g.cu

PMEM_INC = ../../libgpm/include    

INCLUDE := $(CUDA_DIR)/include

SRC   := ./src 
BUILD := ./build
CP_ITER ?= 10

all: start
	$(MAKE) build CP_ITER=10;

generate_input: 
	cd data/; \
	./hotspotex; \
	./hotspotver; \
	cd ..;

build: ${BUILD}/hotspot_${CP_ITER}_gpm #${BUILD}/hotspot_${CP_ITER}_gpm_ndp ${BUILD}/hotspot_${CP_ITER}_fs_gpu ${BUILD}/hotspot_${CP_ITER}_mm_gpu ${BUILD}/hotspot_${CP_ITER}_gpm_rest #${BUILD}/hotspot_${CP_ITER}_volatile  ${BUILD}/hotspot_${CP_ITER}_gpm_near ${BUILD}/hotspot_${CP_ITER}_gpm_debug 

TYPES ?= _gpm _mm_gpu _fs_gpu _gpm_ndp _gpm_rest

run_all:
	make generate_input;
	make all CP_ITER=${CP_ITER};
	mkdir -p ../results;
	for i in ${TYPES}; do \
		sudo PMEM_THREADS=32 ${BUILD}/hotspot_${CP_ITER}$$i > ../results/HS_${CP_ITER}$$i.txt; \
	done


start:
	mkdir -p ${BUILD}

${BUILD}/hotspot_${CP_ITER}_gpm: ./src/hotspot.cu
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $+ -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_CPU -DCP_ITER=${CP_ITER}

${BUILD}/hotspot_${CP_ITER}_gpm_rest: ./src/hotspot.cu
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $+ -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_CPU -DRESTORE_FLAG -DCP_ITER=${CP_ITER}

${BUILD}/hotspot_${CP_ITER}_gpm_debug: ./src/hotspot.cu
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $+ -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_CPU -DCP_ITER=${CP_ITER} -DOUTPUT_NVM_DETAILS

${BUILD}/hotspot_${CP_ITER}_gpm_ndp: ./src/hotspot.cu
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $+ -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_CPU -DGPM_WDP -DCP_ITER=${CP_ITER}

${BUILD}/hotspot_${CP_ITER}_fs_gpu: ./src/hotspot_fs.cu
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $+ -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_GPU -DCP_ITER=${CP_ITER}

${BUILD}/hotspot_${CP_ITER}_mm_gpu: ./src/hotspot_mm.cu
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $+ -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_GPU -DCP_ITER=${CP_ITER}

${BUILD}/hotspot_${CP_ITER}_gpm_near: ./src/hotspot.cu
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $+ -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_GPU -DCP_ITER=1 -DFAKE_NVM -DEMULATE_NVM 

${BUILD}/hotspot_${CP_ITER}_volatile: ./src/hotspot.cu
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $+ -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_CPU -DCP_ITER=${CP_ITER} -DFAKE_NVM

enum: ./src/hotspot.cu 
	$(CC) $(KERNEL_DIM) -deviceemu $(SRC)/hotspot.cu -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debug: ./src/hotspot.cu
	$(CC) $(KERNEL_DIM) -g $(SRC)/hotspot.cu -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debugenum: ./src/hotspot.cu 
	$(CC) $(KERNEL_DIM) -g -deviceemu $(SRC)/hotspot.cu -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean:
	rm -f $(EXE) $(EXE).linkinfo result.txt
	rm -f ${BUILD}/*
