CC = g++

CFLAGS = -Wall
#-D__USE_GNU -D_GNU_SOURCE 
LIBS = -lrt -lpthread -lpmem

#OPENSSL_DIR = ../openssl-1.0.1c
#ifneq ($(OPENSSL_DIR), )
#	OPENSSL_LIBDIR = -L$(OPENSSL_DIR)/lib64
#	OPENSSL_INCDIR = -I$(OPENSSL_DIR)/include
#endif

################################################################
# OS Name (Linux or Darwin)
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# Flags to detect 32-bit or 64-bit OS platform
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")

# These flags will override any settings
ifeq ($(i386),1)
	OS_SIZE = 32
	OS_ARCH = i686
endif

ifeq ($(x86_64),1)
	OS_SIZE = 64
	OS_ARCH = x86_64
endif

# Flags to detect either a Linux system (linux) or Mac OSX (darwin)
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))

# Location of the CUDA Toolkit binaries and libraries
CUDA_PATH       ?= /usr/local/cuda-11.0
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
ifneq ($(DARWIN),)
  CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
else
  ifeq ($(OS_SIZE),32)
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
  else
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
  endif
endif

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

# OS-specific build flags
ifneq ($(DARWIN),) 
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L $(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -arch $(OS_ARCH) 
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m32
  else
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m64
  endif
endif

# OS-architecture specific flags
NVCCFLAGS := -lpci -lnvidia-ml -lpthread -m64 -arch=sm_70 -lineinfo --cudart shared -Xcompiler -fopenmp

# Debug build flags
ifeq ($(dbg),1)
      CCFLAGS   += -g
      NVCCFLAGS += -g -G
      TARGET    := debug
else
      TARGET    := release
endif


# Common includes and paths for CUDA
NVCCINCLUDES      := -I$(CUDA_INC_PATH) -I../../libgpm/include -I .

################################################################

HEADER_DIR = ../
OBJECT_DIR = objs
TARGET_DIR = .

TARGET_FILE = run
TARGET = $(addprefix $(TARGET_DIR)/, $(TARGET_FILE))

SOURCE_FILES = $(wildcard *.c)
HEADER_FILES = $(wildcard *.h) $(wildcard *.h)
OBJECT_FILE = $(SOURCE_FILES:.c=.o)

OBJECT = $(addprefix $(OBJECT_DIR)/, $(OBJECT_FILE))
BUILD  := ./build
SRC    := ./src

all: start ${BUILD}/kvs_gpm ${BUILD}/kvs_gpm_rest ${BUILD}/kvs_get_set_gpm_ndp ${BUILD}/kvs_fs_gpu ${BUILD}/kvs_mm_gpu ${BUILD}/kvs_gpm_ndp ${BUILD}/kvs_get_set_gpm ${BUILD}/kvs_get_set_mm_gpu  ${BUILD}/kvs_get_set_fs_gpu #${BUILD}/kvs_volatile ${BUILD}/kvs_conv ${BUILD}/kvs_gpm_near  ${BUILD}/kvs_gpm_debug

TYPES ?= _gpm _mm_gpu _fs_gpu _gpm_ndp _gpm_rest

run_all:
	make all;
	mkdir -p ../results;
	for i in ${TYPES}; do \
		sudo PMEM_THREADS=32 ${BUILD}/kvs$$i > ../results/kvs$$i.txt; \
		sudo PMEM_THREADS=32 ${BUILD}/kvs_get_set$$i > ../results/kvs_gs$$i.txt; \
	done

run_rest:
	make all;
	mkdir -p ../results;
	for i in _gpm _gpm_rest; do \
		sudo PMEM_THREADS=32 ${BUILD}/kvs$$i > ../results/kvs$$i.txt; \
	done

start:
	mkdir -p ${BUILD}

${BUILD}/kvs_gpm: ${SRC}/insert_test.cu ${SRC}/gpu_hash.h
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(LIBS) -DNVM_ALLOC_CPU

${BUILD}/kvs_gpm_rest: ${SRC}/insert_test.cu ${SRC}/gpu_hash.h
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(LIBS) -DNVM_ALLOC_CPU -DRESTORE_FLAG

${BUILD}/kvs_get_set_gpm: ${SRC}/get_set_test.cu ${SRC}/gpu_hash.h
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(LIBS) -DNVM_ALLOC_CPU

${BUILD}/kvs_get_set_gpm_rest: ${SRC}/get_set_test.cu ${SRC}/gpu_hash.h
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(LIBS) -DNVM_ALLOC_CPU -DRESTORE_FLAG

${BUILD}/kvs_get_set_mm_gpu: ${SRC}/get_set_test_mm.cu ${SRC}/gpu_hash_mm.h
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(LIBS)

${BUILD}/kvs_get_set_fs_gpu: ${SRC}/get_set_test_fs.cu ${SRC}/gpu_hash_fs.h
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(LIBS)

${BUILD}/kvs_get_set_gpm_ndp: ${SRC}/get_set_test.cu ${SRC}/gpu_hash.h
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(LIBS) -DNVM_ALLOC_CPU -DGPM_WDP

${BUILD}/kvs_gpm_debug: ${SRC}/insert_test.cu ${SRC}/gpu_hash.h
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(LIBS) -DNVM_ALLOC_CPU -DOUTPUT_NVM_DETAILS

${BUILD}/kvs_gpm_ndp: ${SRC}/insert_test.cu ${SRC}/gpu_hash.h
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(LIBS) -DNVM_ALLOC_CPU -DGPM_WDP

${BUILD}/kvs_conv: ${SRC}/insert_test.cu ${SRC}/gpu_hash.h
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(LIBS) -DNVM_ALLOC_CPU -DCONV_LOG

${BUILD}/kvs_fs_gpu: ${SRC}/insert_test_fs.cu ${SRC}/gpu_hash_fs.h
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(LIBS)

${BUILD}/kvs_mm_gpu: ${SRC}/insert_test_mm.cu ${SRC}/gpu_hash_mm.h
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(LIBS)

${BUILD}/kvs_gpm_near: ${SRC}/insert_test.cu ${SRC}/gpu_hash.h
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(LIBS) -DNVM_ALLOC_GPU -DFAKE_NVM

${BUILD}/kvs_volatile: ${SRC}/insert_test.cu ${SRC}/gpu_hash.h
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(LIBS) -DNVM_ALLOC_CPU -DFAKE_NVM

$(OBJECT_DIR):
	mkdir $(OBJECT_DIR)

$(OBJECT_DIR)/%.o : %.c
	$(CC) $(CFLAGS) -I $(HEADER_DIR) -I $(CUDA_INC_PATH) -c $< -o $@

$(OBJECT_DIR)/%.o : %.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) -c $< -o $@

.PHONY : clean
clean:
	rm -rf $(TARGET) $(OBJECT) ./objs/gpu_hash.o
	rm -f ${BUILD}/*
