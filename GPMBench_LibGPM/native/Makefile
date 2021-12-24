apps ?= BFS SRAD PS 

run_all:
	mkdir -p ./results;
	for i in ${apps}; do \
		cd $$i; \
		make run_all; \
		cd ..; \
	done
	
run_fig10:
	mkdir -p ./results;
	for i in ${apps}; do \
		cd $$i; \
		make run_fig_10; \
		cd ..; \
	done

figure_10:
	$(MAKE) run_fig10;
	$(MAKE) out_figure_10;

out_figure_10:
	echo "" > out_figure10.txt;
	for i in ${apps}; do                                            \
		time_gpm=$$(grep "runtime" results/$${i}_gpm.txt | grep -oE "[0-9]+\.[0-9]+");     \
		time_gpm_ndp=$$(grep "runtime" results/$${i}_gpm_ndp.txt | grep -oE "[0-9]+\.[0-9]+");     \
		time_gpm_ndp_persist=$$(grep "Persist" results/$${i}_gpm_ndp.txt | grep -oE "[0-9]+\.[0-9]+");     \
		time_capfs=$$(grep "runtime" results/$${i}_fs_gpu.txt | grep -oE "[0-9]+\.[0-9]+");     \
		time_capmm=$$(grep "runtime" results/$${i}_mm_gpu.txt | grep -oE "[0-9]+\.[0-9]+");     \
		time_capmm_persist=$$(grep "Persist" results/$${i}_mm_gpu.txt | grep -oE "[0-9]+\.[0-9]+");     \
		speedup_gpm_ndp=$$(awk "BEGIN {print $${time_capfs}/$${time_gpm_ndp}}"); \
		speedup_gpm=$$(awk "BEGIN {print $${time_capfs}/$${time_gpm}}"); \
		speedup_gpm_eadr=$$(awk "BEGIN {print $${time_capfs}/($${time_gpm_ndp}-$${time_gpm_ndp_persist})}"); \
		speedup_cap_eadr=$$(awk "BEGIN {print $${time_capfs}/($${time_capmm}-$${time_capmm_persist})}"); \
		echo "$$i\t$${speedup_gpm_ndp}\t$${speedup_gpm}\t$${speedup_gpm_eadr}\t$${speedup_cap_eadr}" >> out_figure10.txt; \
	done
	cat out_figure10.txt


out_figure_9:
	./run.sh; cat out_figure9.txt;
	
figure_9:
	$(MAKE) run_all;
	$(MAKE) out_figure_9;
