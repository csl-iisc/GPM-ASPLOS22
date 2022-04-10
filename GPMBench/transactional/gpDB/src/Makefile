
# set config to debug, release
CONFIG = release

# file with main function
ENTRY_POINT = 

# custom compiler declarations
#CUSTOM_FLAGS = -D VIRG_DEBUG #-D VIRG_NOPINNED #-D VIRG_DEBUG_SLOTS

CUDA = /usr/local/cuda-11.0
LINK_FLAGS = -lm -lpthread -L$(CUDA_LIBRARY) -lcuda -lcudart -lpci
NVCC = $(CUDA)/bin/nvcc -Xcompiler -fopenmp -std=c++11 -lnvidia-ml -lpmem -arch=sm_70 -DNVM_ALLOC_CPU
GCC =  $(NVCC)#/usr/bin/gcc 
GPP =  $(NVCC)#/usr/bin/g++
CUDA_INCLUDE = $(CUDA)/include
CUDA_LIBRARY = $(CUDA)/lib64
GTEST = ../gtest

FLAGS = #-Wall -Wextra -std=c99

COMMON = ../../../libgpm
INCLUDE_FLAGS = -I $(CUDA_INCLUDE) -I . -I sql/ -I$(COMMON)/include
DEBUG_FLAGS =
RELEASE_FLAGS = -O2 
FLAG = $(FLAGS) $(CUSTOM_FLAGS) $(INCLUDE_FLAGS)
CPPFLAG = $(CUSTOM_FLAGS) $(INCLUDE_FLAGS)

ifeq ($(CONFIG),debug)
	OF = debug
	CC = $(GCC)
	COMPILE_FLAGS = $(FLAG) $(DEBUG_FLAGS)
endif
ifeq ($(CONFIG),release)
	OF = release
	CC = $(GCC)
	COMPILE_FLAGS = $(FLAG) $(RELEASE_FLAGS)
endif

SPECIALS = opcodelist.cpp sql/lex.yy.cpp sql/sql.tab.cpp
CFILES = $(filter-out $(SPECIALS),$(wildcard *.cpp) $(wildcard */*.cpp) $(ENTRY_POINT)) $(SPECIALS)
OFILES = $(patsubst %.cpp,../$(OF)/%.o,$(CFILES))
LFILES = $(OFILES) ../$(OF)/vm/virginia-single.o ../$(OF)/vm/vm_gpu.o ../$(OF)/table/insert.o ../$(OF)/table/update.o

TESTCFILES = #$(wildcard test/*.cc)
TESTOFILES = #$(patsubst %.cc,../$(OF)/%.o,$(TESTCFILES))

virginian: ../lib/virginian.a $(ENTRY_POINT) test ../db/comparedb 
	#../$(OF)/virginiantest 
	#cd ../$(OF) && ./virginiantest
	#cd ../$(OF) && valgrind --leak-check=full --suppressions=/home/bakks/virginian/virginian/src/cuda.supp ./virginiantest
	#  --gen-suppressions=yes

clean:
	rm -rf ../debug/* ../release/* opcodelist.cpp vm_gpu.ptx sql/lex.yy.cpp sql/sql.tab.*
	rm -rf ../lib/virginian.*
	mkdir -p ../debug ../release/
	ls -d */ | sed 's/[a-z]*\//..\/debug\/\0/' | xargs mkdir
	ls -d */ | sed 's/[a-z]*\//..\/release\/\0/' | xargs mkdir

count:
	wc -l *.cpp *.h */*.cpp */*.cu */*.h */*.cc */*.y */*.l

$(OFILES): ../$(OF)/%.o: %.cpp virginian.h
	$(CC) $(COMPILE_FLAGS) -dc $< -o $@

../$(OF)/vm/virginia-single.o: vm/virginia.cpp virginian.h
	$(CC) -I. -dc -D __SINGLE $< -o $@


../$(OF)/db/create.o: db/create.cpp virginian.h
	$(CC) $(COMPILE_FLAGS)  -x cu -dc $< -o $@

../$(OF)/db/close.o: db/close.cpp virginian.h
	$(CC) $(COMPILE_FLAGS)  -x cu -dc $< -o $@

../$(OF)/db/write.o: db/write.cpp virginian.h
	$(CC) $(COMPILE_FLAGS)  -x cu -dc $< -o $@

../$(OF)/db/open.o: db/open.cpp virginian.h
	$(CC) $(COMPILE_FLAGS)  -x cu -dc $< -o $@

../$(OF)/vm/vm_gpu.o: vm/vm_gpu.cu virginian.h
	$(NVCC) $(CUSTOM_FLAGS) $(INCLUDE_FLAGS) -O2 -dc $< -o $@

../$(OF)/libgpm.o: $(COMMON)/src/libgpm.cu
	$(NVCC) $(CUSTOM_FLAGS) $(INCLUDE_FLAGS) -O2 -dc $< -o $@
	
../$(OF)/gpm-helper.o: $(COMMON)/src/gpm-helper.cu
	$(NVCC) $(CUSTOM_FLAGS) $(INCLUDE_FLAGS) -O2 -dc $< -o $@
	
../$(OF)/table/insert.o: table/insert.cu virginian.h
	$(NVCC) $(CUSTOM_FLAGS) $(INCLUDE_FLAGS) -x cu -O2 -dc $< -o $@
	
../$(OF)/table/update.o: table/update.cu virginian.h
	$(NVCC) $(CUSTOM_FLAGS) $(INCLUDE_FLAGS) -x cu -O2 -dc $< -o $@


opcodelist.cpp: opcodelist.awk virginian.h
	./opcodelist.awk virginian.h > opcodelist.cpp

../$(OF)/virginian: $(LFILES)
	$(CC) $(LFILES) $(LINK_FLAGS) -o ../$(OF)/virginian

../lib/virginian.a: $(LFILES)
	ar rvs ../lib/virginian.a $(LFILES)
	cp virginian.h ../lib

../db/comparedb:
	make -s -C ../db clean
	make -s -C ../db
	../db/generate comparedb 100000000

ptx: vm/vm_gpu.cu
	$(NVCC) $(CUSTOM_FLAGS) $(INCLUDE_FLAGS) --ptxas-options="-v" -ptx $< -o vm_gpu.ptx

$(TESTOFILES): ../$(OF)/%.o: %.cc virginian.h test/test.h
	$(NVCC) -I/usr/local/lib $(CPPFLAG) -c $< -o $@

../$(OF)/virginiantest: $(LFILES) $(TESTOFILES) ../lib/virginian.a
	$(NVCC) $(LINK_FLAGS) $(TESTOFILES) ../lib/virginian.a ../lib/libgtest.a -lstdc++ -lpthread -L/usr/local/lib -lgsl -lgslcblas -o ../$(OF)/virginiantest

sql/sql.tab.cpp: sql/sql.y sql/node.h
	bison --defines=sql/sql.tab.h -o sql/sql.tab.cpp sql/sql.y

sql/sql.tab.h: sql/sql.y sql/node.h
	bison --defines=sql/sql.tab.h -o sql/sql.tab.cpp sql/sql.y

sql/lex.yy.cpp: sql/sql.l sql/sql.tab.h sql/node.h
	flex -o sql/lex.yy.cpp sql/sql.l





.PHONY: virginian
.PHONY: clean
.PHONY: count

