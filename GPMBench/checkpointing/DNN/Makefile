CC = nvcc 
CC_FLAGS = -rdc=true -O3 -lpthread -lnvidia-ml -arch=sm_75 -lcudnn -lcudadevrt -lcudart_static -lrt -ldl -lcublas -lpci -lpmem -Xcompiler -fopenmp
CUDA_SDK_PATH := $(SDK_DIR)
CUDA_PATH := /usr/local/cuda-11.0


INCLUDES := -I$(CUDA_PATH)/include -I../../libgpm/include

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

build: ${BUILD}/lenet_${CP_ITER}_gpm ${BUILD}/lenet_${CP_ITER}_gpm_ndp ${BUILD}/lenet_${CP_ITER}_fs_gpu ${BUILD}/lenet_${CP_ITER}_mm_gpu ${BUILD}/lenet_${CP_ITER}_gpm_rest #${BUILD}/lenet_${CP_ITER}_volatile ${BUILD}/lenet_${CP_ITER}_gpm_near ${BUILD}/lenet_${CP_ITER}_gpm_debug 

TYPES ?= _gpm _mm_gpu _fs_gpu _gpm_ndp _gpm_rest _gpufs 

run_all:
	make all CP_ITER=${CP_ITER};
	mkdir -p ../results;
	for i in ${TYPES}; do \
		sudo PMEM_THREADS=32 ${BUILD}/lenet_${CP_ITER}$$i > ../results/DNN_${CP_ITER}$$i.txt; \
	done

run_fig_10: 
	sudo ${BUILD}/lenet_${CP_ITER}_gpm_ndp > ../results/DNN_${CP_ITER}_gpm_ndp.txt;


start:
	mkdir -p ${BUILD}

${BUILD}/lenet_${CP_ITER}_gpm: ${SRC}/lenet.cu ${SRC}/readubyte.cpp
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $^ -o $@ -I$(CUDA_SDK_PATH)/common/inc $(INCLUDES) -L$(CUDA_DIR)/lib $(CUTIL_LIB) -DCP_ITER=${CP_ITER} -DNVM_ALLOC_CPU -std=c++11

${BUILD}/lenet_${CP_ITER}_gpm_rest: ${SRC}/lenet.cu ${SRC}/readubyte.cpp
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $^ -o $@ -I$(CUDA_SDK_PATH)/common/inc $(INCLUDES) -L$(CUDA_DIR)/lib $(CUTIL_LIB) -DCP_ITER=${CP_ITER} -DNVM_ALLOC_CPU -DRESTORE_FLAG -std=c++11

${BUILD}/lenet_${CP_ITER}_gpm_debug: ${SRC}/lenet.cu ${SRC}/readubyte.cpp
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $^ -o $@ -I$(CUDA_SDK_PATH)/common/inc $(INCLUDES) -L$(CUDA_DIR)/lib $(CUTIL_LIB) -DCP_ITER=${CP_ITER} -DNVM_ALLOC_CPU -std=c++11 -DOUTPUT_NVM_DETAILS

${BUILD}/lenet_${CP_ITER}_gpm_ndp: ${SRC}/lenet.cu ${SRC}/readubyte.cpp
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $^ -o $@ -I$(CUDA_SDK_PATH)/common/inc $(INCLUDES) -L$(CUDA_DIR)/lib $(CUTIL_LIB) -DCP_ITER=${CP_ITER} -DGPM_WDP -DNVM_ALLOC_CPU -std=c++11

${BUILD}/lenet_${CP_ITER}_mm_gpu: ${SRC}/lenet_mm.cu ${SRC}/readubyte.cpp
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $^ -o $@ -I$(CUDA_SDK_PATH)/common/inc $(INCLUDES) -L$(CUDA_DIR)/lib $(CUTIL_LIB) -DCP_ITER=${CP_ITER}

${BUILD}/lenet_${CP_ITER}_fs_gpu: ${SRC}/lenet_fs.cu ${SRC}/readubyte.cpp
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $^ -o $@ -I$(CUDA_SDK_PATH)/common/inc $(INCLUDES) -L$(CUDA_DIR)/lib $(CUTIL_LIB) -DCP_ITER=${CP_ITER}

${BUILD}/lenet_${CP_ITER}_gpm_near: ${SRC}/lenet.cu ${SRC}/readubyte.cpp
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $^ -o $@ -I$(CUDA_SDK_PATH)/common/inc $(INCLUDES) -L$(CUDA_DIR)/lib $(CUTIL_LIB) -DCP_ITER=${CP_ITER} -DNVM_ALLOC_GPU -DFAKE_NVM -DEMULATE_NVM -std=c++11

${BUILD}/lenet_${CP_ITER}_volatile: ${SRC}/lenet.cu ${SRC}/readubyte.cpp
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $^ -o $@ -I$(CUDA_SDK_PATH)/common/inc $(INCLUDES) -L$(CUDA_DIR)/lib $(CUTIL_LIB) -DCP_ITER=${CP_ITER} -DNVM_ALLOC_CPU -std=c++11 -DFAKE_NVM

clean:
	rm -f *.linkinfo
	mv ${BUILD}/lenet_10_gpufs . 
	rm -rf ${BUILD}/ #!("lenet_10_gpufs")
	mkdir -p ${BUILD}
	mv lenet_10_gpufs ${BUILD}/
