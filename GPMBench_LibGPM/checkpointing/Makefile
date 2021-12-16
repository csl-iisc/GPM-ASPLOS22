apps ?= HS DNN CFD BLK 
CP_ITERS ?= 10

run_all:
	for i in ${apps}; do \
		cd $$i; \
		make run_all CP_ITERS=${CP_ITERS}; \
		cd ..; \
	done
	
out_figure_9:
	rm -f out_figure9.txt; 
	./run.sh;
	cat out_figure9.txt;
	
	
out_table_5:
	echo "" > out_table5.txt; 
	for i in ${apps}; do                                            \
		time_gpm=$$(grep "CheckpointTime" results/$${i}_${CP_ITERS}_gpm.txt | grep -oE "[0-9]+\.[0-9]+");     \
		time_recov=$$(grep "Recovery" results/$${i}_${CP_ITERS}_gpm_rest.txt | grep -oE "[0-9]+\.[0-9]+");     \
		time=$$(awk "BEGIN {print $${time_recov}/$${time_gpm}}"); \
		echo "$$i\t$${time}" >> out_table5.txt; \
	done
	cat out_table5.txt;
	
out_figure_12: 
	./extract_bw.sh; cat out_figure12.txt;
	
figure_9:
	$(MAKE) run_all; #TYPES="_gpm _fs_gpu _mm_gpu";
	$(MAKE) out_figure_9;
	
table_5:
	$(MAKE) run_all TYPES="_gpm _gpm_rest";
	$(MAKE) out_table_5;	
