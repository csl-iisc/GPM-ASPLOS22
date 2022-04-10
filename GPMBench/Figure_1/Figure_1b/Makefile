APPS   		 ?= bfs srad ps
RUNS         ?= 1
MAKE         ?= make

all:
	make run
	make out_figure_1b

run:
	for i in ${APPS}; do \
		cd $$i;  \
		sudo $(MAKE) run; \
		cd ..;       \
	done

out_figure1b: 
	./run.sh

