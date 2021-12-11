# 
#  Copyright (c) 2016 University of Cordoba and University of Illinois
#  All rights reserved.
# 
#  Developed by:    IMPACT Research Group
#                   University of Cordoba and University of Illinois
#                   http://impact.crhc.illinois.edu/
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  with the Software without restriction, including without limitation the 
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#  sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#       > Redistributions of source code must retain the above copyright notice,
#         this list of conditions and the following disclaimers.
#       > Redistributions in binary form must reproduce the above copyright
#         notice, this list of conditions and the following disclaimers in the
#         documentation and/or other materials provided with the distribution.
#       > Neither the names of IMPACT Research Group, University of Cordoba, 
#         University of Illinois nor the names of its contributors may be used 
#         to endorse or promote products derived from this Software without 
#         specific prior written permission.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
#  CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
#  THE SOFTWARE.
# 

CXX=/usr/local/cuda-11.0/bin/nvcc
CXX_FLAGS=-lpci -lpmem -rdc=true -std=c++11 -lpthread -lnvidia-ml -O3 -arch=sm_70
CHAI_CUDA_LIB=/usr/local/cuda-11.0/lib64
CHAI_CUDA_INC=/usr/local/cuda-11.0/include
PMEM_LIB = ../../libgpm/src/gpm-helper.cu ../../libgpm/src/libgpm.cu ../../libgpm/src/libgpmlog.cu ../../libgpm/src/libgpmcp.cu
PMEM_INC = ../../libgpm/include

BUILD ?= ./build

TYPES ?= _gpm _fs_gpu _mm_gpu 

run_all: 
	make all;
	mkdir -p ../results;
	for i in ${TYPES}; do sudo PMEM_THREADS=32 ${BUILD}/BFS$$i > ../results/BFS$$i.txt; done

ifndef CHAI_CUDA_LIB
    $(error CHAI_CUDA_LIB not defined. This environment variable must be defined to point to the location of the CUDA library)
endif
LIB=-L/usr/lib/ -L$(CHAI_CUDA_LIB) -lm 
ifndef CHAI_CUDA_INC
    $(error CHAI_CUDA_INC not defined. This environment variable must be defined to point to the location of the CUDA header files)
endif
INC=-I$(CHAI_CUDA_INC) -I$(PMEM_INC)

DEP=./src/kernel.h ./src/main.cu ./src/kernel.cu support/libgpm.h support/cuda-setup.h support/timer.h support/verify.h #kernel_cpu.cu 
SRC    := ./src/main.cu ./src/kernel.cu #kernel_cpu.cu 
SRC_MM := ./src/main_mm.cu ./src/kernel_mm.cu 
SRC_FS := ./src/main_fs.cu ./src/kernel_fs.cu 
EXE    := run
BUILD  := ./build

all: 
	$(MAKE) start
	$(MAKE) build; 

run_fig_10: 
	sudo PMEM_THREADS=32 ${BUILD}/BFS_gpm_ndp > ../results/BFS_gpm_ndp.txt;


start:
	mkdir -p ${BUILD}
	mkdir -p input/ 
	#./process_input.sh

build: ${BUILD}/BFS_mm_gpu ${BUILD}/BFS_fs_gpu ${BUILD}/BFS_gpm ${BUILD}/BFS_gpm_ndp #${BUILD}/BFS_volatile ${BUILD}/BFS_gpm_near ${BUILD}/BFS_gpm_debug ${BUILD}/BFS_pure_cpu

${BUILD}/BFS_mm_gpu: ./src/main_mm.cu ./src/kernel_mm.cu
	$(CXX) $(CXX_FLAGS) $(SRC_MM) $(INC) -o $@

${BUILD}/BFS_pure_cpu: ./src/main_cpu.cu
	$(CXX) $(CXX_FLAGS) $^ $(INC) -o $@

${BUILD}/BFS_fs_gpu: $(SRC_FS)
	$(CXX) $(CXX_FLAGS) $(SRC_FS) $(INC) -o $@

${BUILD}/BFS_gpm: $(SRC)
	$(CXX) -DNVM_ALLOC_CPU $(CXX_FLAGS) $(SRC) $(INC) -o $@

${BUILD}/BFS_gpm_debug: $(SRC)
	$(CXX) -DNVM_ALLOC_CPU $(CXX_FLAGS) $(SRC) $(INC) -o $@ -DOUTPUT_NVM_DETAILS

${BUILD}/BFS_gpm_ndp: $(SRC)
	$(CXX) -DNVM_ALLOC_CPU $(CXX_FLAGS) $(SRC) $(INC) -o $@ -DGPM_WDP

${BUILD}/BFS_gpm_near: $(SRC)
	$(CXX)  -DNVM_ALLOC_GPU -DFAKE_NVM $(CXX_FLAGS) $(SRC) $(INC) -o $@ 

${BUILD}/BFS_volatile: $(SRC)
	$(CXX) -DNVM_ALLOC_CPU -DFAKE_NVM $(CXX_FLAGS) $(SRC) $(INC) -o $@ 

clean:
	rm -f ${BUILD}/*

clobber: clean
