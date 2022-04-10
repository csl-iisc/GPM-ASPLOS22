APPS         ?= gpkvs pmemkv matrixkv rocksdb 
RUNS         ?= 1
OPERATTIONS  ?= 52428800
MAKE         ?= make

all:
	make run 
	make out_figure_1a

run:
	for i in ${APPS} ; do cd $$i ; sudo $(MAKE) run ; cd .. ; \
	done

out_figure_1a: 
	./run.sh

