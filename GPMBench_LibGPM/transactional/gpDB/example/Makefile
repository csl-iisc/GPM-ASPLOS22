
LIB = ../lib
CUDA = /usr/local/cuda-10.2

all: test.c
	$(CUDA)/bin/nvcc -I$(LIB) -I$(CUDA)/include -L$(CUDA)/lib -lcudart test.c $(LIB)/virginian.a -o test
	./test

clean:
	rm -rf test

