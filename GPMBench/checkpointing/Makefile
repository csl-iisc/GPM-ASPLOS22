apps ?= DNN CFD BLK HS 
CP_ITERS ?= 10

run_all:
	mkdir -p ./results;
	for i in ${apps}; do \
		cd $$i; \
		make run_all CP_ITERS=${CP_ITERS}; \
		cd ..; \
	done
	
run_fig10:
	mkdir -p ./results;
	for i in ${apps}; do \
		cd $$i; \
		make run_fig_10 CP_ITERS=${CP_ITERS}; \
		cd ..; \
	done
	

out_figure_9:
	mkdir -p ./results;
	rm -f out_figure9.txt; 
	./run.sh;
	cat out_figure9.txt;
	
out_figure_10:
	echo "" > out_figure10.txt;
	for i in ${apps}; do                                            \
		time_gpm=$$(grep "CheckpointTime" results/$${i}_${CP_ITERS}_gpm.txt | grep -oE "[0-9]+\.[0-9]+");     \
		time_gpm_ndp=$$(grep "CheckpointTime" results/$${i}_${CP_ITERS}_gpm_ndp.txt | grep -oE "[0-9]+\.[0-9]+");     \
		time_gpm_ndp_persist=$$(grep "PersistTime" results/$${i}_${CP_ITERS}_gpm_ndp.txt | grep -oE "[0-9]+\.[0-9]+");     \
		time_capfs=$$(grep "CheckpointTime" results/$${i}_${CP_ITERS}_fs_gpu.txt | grep -oE "[0-9]+\.[0-9]+");     \
		time_capmm=$$(grep "CheckpointTime" results/$${i}_${CP_ITERS}_mm_gpu.txt | grep -oE "[0-9]+\.[0-9]+");     \
		time_capmm_persist=$$(grep "PersistTime" results/$${i}_${CP_ITERS}_mm_gpu.txt | grep -oE "[0-9]+\.[0-9]+");     \
		speedup_gpm_ndp=$$(awk "BEGIN {print $${time_capfs}/$${time_gpm_ndp}}"); \
		speedup_gpm=$$(awk "BEGIN {print $${time_capfs}/$${time_gpm}}"); \
		speedup_gpm_eadr=$$(awk "BEGIN {print $${time_capfs}/($${time_gpm_ndp}-$${time_gpm_ndp_persist})}"); \
		speedup_cap_eadr=$$(awk "BEGIN {print $${time_capfs}/($${time_capmm}-$${time_capmm_persist})}"); \
		echo "$$i\t$${speedup_gpm_ndp}\t$${speedup_gpm}\t$${speedup_gpm_eadr}\t$${speedup_cap_eadr}" >> out_figure10.txt; \
	done
	cat out_figure10.txt;

out_table_5:
	echo "" > out_table5.txt; 
	for i in ${apps}; do                                            \
		time_gpm=$$(grep "CheckpointTime" results/$${i}_${CP_ITERS}_gpm.txt | grep -oE "[0-9]+\.[0-9]+");     \
		time_recov=$$(grep "Recovery" results/$${i}_${CP_ITERS}_gpm_rest.txt | grep -oE "[0-9]+\.[0-9]+");     \
		time=$$(awk "BEGIN {print $${time_recov}/$${time_gpm}}"); \
		echo "$$i\t$${time}" >> out_table5.txt; \
	done
	cat out_table5.txt;
	
	
figure_9:
	$(MAKE) run_all; #TYPES="_gpm _fs_gpu _mm_gpu";
	$(MAKE) out_figure_9;
	
table_5:
	$(MAKE) run_all TYPES="_gpm _gpm_rest";
	$(MAKE) out_table_5;	

figure_10:
	$(MAKE) run_fig10;
	$(MAKE) out_figure_10;
