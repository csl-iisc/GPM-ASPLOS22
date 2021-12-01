apps ?= SRAD PS BFS 

run_all:
	for i in ${apps}; do \
		cd $$i; \
		make run_all; \
		cd ..; \
	done
	
out_figure_9:
	./run.sh; cat out_figure9.txt;
	
figure_9:
	$(MAKE) run_all TYPES="_gpm _fs_gpu _mm_gpu";
	$(MAKE) out_figure_9;
