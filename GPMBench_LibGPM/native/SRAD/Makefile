# CUDA toolkit installation path
CUDA_DIR = /usr/local/cuda-11.0
 
# CUDA toolkit libraries
CUDA_LIB_DIR := $(CUDA_DIR)/lib64
ifeq ($(shell uname -m), x86_64)
     ifeq ($(shell if test -d $(CUDA_DIR)/lib64; then echo T; else echo F; fi), T)
        CUDA_LIB_DIR := $(CUDA_DIR)/lib64
     endif
endif
 
SDK_DIR = /usr/local/cuda-11.0/samples/

OPENCL_DIR = /if10/kw5na/Packages/AMD-APP-SDK-v2.8-RC-lnx64
OPENCL_INC = $(OPENCL_DIR)/include/
OPENCL_LIB = $(OPENCL_DIR)/lib/x86_64/ -lOpenCL

CC := $(CUDA_DIR)/bin/nvcc

NVCC_FLAGS := -lpmem -rdc=true -std=c++11 -lpthread -lnvidia-ml -O3 -arch=sm_70 -lpci -Xcompiler -fopenmp

PMEM_LIB := ../../libgpm/src/pmem-helper.cu ../../libgpm/src/libpmem-g.cu ../../libgpm/src/libpmemlog-g. cu ../../libgpm/src/libpmemcp-g.cu

PMEM_INC = ../../libgpm/include    

INCLUDE := $(CUDA_DIR)/include

SRC = ./src 
BUILD = ./build

#EXE = srad

all: start ${BUILD}/SRAD_fs_gpu ${BUILD}/SRAD_mm_gpu ${BUILD}/SRAD_gpm ${BUILD}/SRAD_gpm_ndp # ${BUILD}/srad_gpm_near ${BUILD}/srad_volatile ${BUILD}/srad_gpm_debug ${BUILD}/srad_pure_cpu

TYPES ?= _gpm _fs_gpu _mm_gpu _gpufs

run_all: 
	make all;
	mkdir -p ../results;
	for i in ${TYPES}; do sudo PMEM_THREADS=32 ${BUILD}/SRAD$$i 131072 1024 0 31 0 31 0.5 50 > ../results/SRAD$$i.txt; done

run_fig_10: 
	make all;
	sudo ${BUILD}/SRAD_gpm_ndp 131072 1024 0 31 0 31 0.5 50 > ../results/SRAD_gpm_ndp.txt;

start:
	mkdir -p ${BUILD}

${BUILD}/SRAD_mm_gpu: ./src/srad_mm.cu ./src/srad_kernel_mm.cu
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $< -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_GPU

${BUILD}/SRAD_fs_gpu: ./src/srad_fs.cu ./src/srad_kernel_fs.cu
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $< -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_GPU

${BUILD}/SRAD_gpm: ./src/srad.cu ./src/srad_kernel.cu
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $< -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_CPU -DEMULATE_NVM

${BUILD}/srad_pure_cpu: ./src/srad_cpu.cu
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $< -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_CPU -DEMULATE_NVM

${BUILD}/SRAD_gpm_ndp: ./src/srad.cu ./src/srad_kernel.cu
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $< -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_CPU -DEMULATE_NVM -DGPM_WDP

${BUILD}/srad_volatile: ./src/srad.cu ./src/srad_kernel.cu
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $< -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_CPU -DFAKE_NVM 

${BUILD}/srad_gpm_near: ./src/srad.cu ./src/srad_kernel.cu
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $< -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_GPU -DFAKE_NVM -DEMULATE_NVM

${BUILD}/srad_gpm_debug: ./src/srad.cu ./src/srad_kernel.cu
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $< -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_CPU -DEMULATE_NVM -DOUTPUT_NVM_DETAILS

enum: ${SRC}/srad.cu
	$(CC) $(KERNEL_DIM) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debug: ${SRC}/srad.cu
	$(CC) $(KERNEL_DIM) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debugenum: ${SRC}/srad.cu
	$(CC) $(KERNEL_DIM) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean:
	rm -f *.linkinfo
	mv ${BUILD}/SRAD_gpufs . 
	rm -rf ${BUILD}/ #!("SRAD_10_gpufs")
	mkdir -p ${BUILD}
	mv SRAD_gpufs ${BUILD}/
