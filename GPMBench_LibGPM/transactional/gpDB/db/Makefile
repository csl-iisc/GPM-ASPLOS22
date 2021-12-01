CUDA = /usr/local/cuda-11.0
NVCC = $(CUDA)/bin/nvcc

generate: generate.cu ../lib/virginian.a ../src/virginian.h
	$(NVCC)  -Xcompiler -fopenmp -O2 -lnvidia-ml -lpmem -lpci -I ../src -I ../lib -I /usr/local/cuda/include -I ../../../libgpm/include generate.cu -o generate.o -dc
	$(NVCC)  -Xcompiler -fopenmp generate.o  -lpmem -lpci -lnvidia-ml ../lib/virginian.a -lcudart -L /usr/local/lib -L /usr/local/cuda/lib -o generate

../lib/virginian.a: ../src/Makefile
	make -C ../src ../lib/virginian.a

clean:
	rm -f generate generate.o

.PHONY: clean

