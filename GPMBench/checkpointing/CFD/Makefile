CUDA_DIR = /usr/local/cuda-11.0
 
CUDA_LIB_DIR := $(CUDA_DIR)/lib
ifeq ($(shell uname -m), x86_64)
     ifeq ($(shell if test -d $(CUDA_DIR)/lib64; then echo T; else echo F; fi), T)
        CUDA_LIB_DIR := $(CUDA_DIR)/lib64
     endif
endif
 
# CUDA SDK installation path
SDK_DIR = $(CUDA_DIR)/samples
 
CC = nvcc
CC_FLAGS = -O3 -lpthread -lnvidia-ml -lpmem -arch=sm_70 -lpci -Xcompiler -fopenmp
CUDA_SDK_PATH := $(SDK_DIR)
PMEM_INC = ../../libgpm/include
PMEM_LIB = ../../common/src/gpm-helper.cu ../../common/src/libgpm.cu ../../common/src/libgpmlog.cu ../../common/src/libgpmcp.cu

# Determine the correct version of the cutil library
CUTIL_LIB =  # -lcutil
ifeq ($(shell uname -m), x86_64)
     ifeq ($(shell if test -e $(SDK_DIR)/lib/libcutil_x86_64.a; then echo T; else echo F; fi), T)
        CUTIL_LIB = #-lcutil_x86_64
     endif
endif

BUILD := ./build
CP_ITER ?= 10
SRC := ./src

all: start
	#$(MAKE) build CP_ITER=5;
	$(MAKE) build CP_ITER=10;
	
TYPES ?= _gpm _mm_gpu _fs_gpu _gpm_ndp _gpm_rest _gpufs

run_all:
	make all CP_ITER=${CP_ITER};
	mkdir -p ../results;
	for i in ${TYPES}; do \
		sudo PMEM_THREADS=32 ${BUILD}/CFD_${CP_ITER}$$i data/missile.domn.0.2M > ../results/CFD_${CP_ITER}$$i.txt; \
	done

run_fig_10:
	sudo ${BUILD}/CFD_${CP_ITER}_gpm_ndp data/missile.domn.0.2M > ../results/CFD_${CP_ITER}_gpm_ndp.txt;

build:  ${BUILD}/CFD_${CP_ITER}_gpm ${BUILD}/CFD_${CP_ITER}_gpm_ndp ${BUILD}/CFD_${CP_ITER}_fs_gpu ${BUILD}/CFD_${CP_ITER}_mm_gpu ${BUILD}/CFD_${CP_ITER}_gpm_rest #${BUILD}/CFD_${CP_ITER}_gpm_debug ${BUILD}/CFD_${CP_ITER}_volatile ${BUILD}/CFD_${CP_ITER}_gpm_near #${BUILD}/CFD_${CP_ITER}_mm_tx_gpu

start:
	mkdir -p ${BUILD}

${BUILD}/CFD_${CP_ITER}_gpm: ${SRC}/CFD.cu
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $? -o $@ -I$(CUDA_SDK_PATH)/common/inc -I$(PMEM_INC) -L$(CUDA_DIR)/lib $(CUTIL_LIB) -DNVM_ALLOC_CPU -DCP_ITER=${CP_ITER}

${BUILD}/CFD_${CP_ITER}_gpm_rest: ${SRC}/CFD.cu
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $? -o $@ -I$(CUDA_SDK_PATH)/common/inc -I$(PMEM_INC) -L$(CUDA_DIR)/lib $(CUTIL_LIB) -DNVM_ALLOC_CPU -DRESTORE_FLAG -DCP_ITER=${CP_ITER}

${BUILD}/CFD_${CP_ITER}_gpm_debug: ${SRC}/CFD.cu
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $? -o $@ -I$(CUDA_SDK_PATH)/common/inc -I$(PMEM_INC) -L$(CUDA_DIR)/lib $(CUTIL_LIB) -DNVM_ALLOC_CPU -DCP_ITER=${CP_ITER} -DOUTPUT_NVM_DETAILS

${BUILD}/CFD_${CP_ITER}_gpm_ndp: ${SRC}/CFD.cu
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $? -o $@ -I$(CUDA_SDK_PATH)/common/inc -I$(PMEM_INC) -L$(CUDA_DIR)/lib $(CUTIL_LIB) -DNVM_ALLOC_CPU -DGPM_WDP -DCP_ITER=${CP_ITER}

${BUILD}/CFD_${CP_ITER}_fs_gpu: ${SRC}/CFD_fs.cu
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $? -o $@ -I$(CUDA_SDK_PATH)/common/inc -I$(PMEM_INC) -L$(CUDA_DIR)/lib $(CUTIL_LIB) -DCP_ITER=${CP_ITER}

${BUILD}/CFD_${CP_ITER}_mm_gpu: ${SRC}/CFD_mm.cu
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $? -o $@ -I$(CUDA_SDK_PATH)/common/inc -I$(PMEM_INC) -L$(CUDA_DIR)/lib $(CUTIL_LIB) -DCP_ITER=${CP_ITER}

${BUILD}/CFD_${CP_ITER}_gpm_near: ${SRC}/CFD.cu
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $? -o $@ -I$(CUDA_SDK_PATH)/common/inc -I$(PMEM_INC) -L$(CUDA_DIR)/lib $(CUTIL_LIB) -DCP_ITER=${CP_ITER} -DNVM_ALLOC_GPU -DFAKE_NVM -DEMULATE_NVM

${BUILD}/CFD_${CP_ITER}_volatile: ${SRC}/CFD.cu
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $? -o $@ -I$(CUDA_SDK_PATH)/common/inc -I$(PMEM_INC) -L$(CUDA_DIR)/lib $(CUTIL_LIB) -DCP_ITER=${CP_ITER} -DNVM_ALLOC_CPU -DFAKE_NVM
clean:
	rm -f *.linkinfo
	mv ${BUILD}/CFD_10_gpufs . 
	rm -rf ${BUILD}/ #!("lenet_10_gpufs")
	mkdir -p ${BUILD}
	mv CFD_10_gpufs ${BUILD}/
