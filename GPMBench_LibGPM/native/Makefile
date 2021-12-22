apps ?= SRAD PS BFS 

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

out_figure_10:
	echo "" > out_figure10.txt;
	for i in ${apps}; do                                            \
		time_gpm_ndp=$$(grep "runtime" results/$${i}_gpm_ndp.txt | grep -oE "[0-9]+\.[0-9]+");     \
		time_gpm_ndp_persist=$$(grep "Persist" results/$${i}_gpm_ndp.txt | grep -oE "[0-9]+\.[0-9]+");     \
		time_capfs_persist=$$(grep "Persist" results/$${i}_mm_gpu.txt | grep -oE "[0-9]+\.[0-9]+");     \
		time_capfs=$$(grep "runtime" results/$${i}_fs_gpu.txt | grep -oE "[0-9]+\.[0-9]+");     \
		speedup_gpm_ndp=$$(awk "BEGIN {print $${time_capfs}/$${time_gpm_ndp}}"); \
		echo "$$i\t$${time_gpm_ndp}\t$${time_gpm_ndp_persist}\t$${speedup_gpm_ndp}\t$${time_capfs_persist}" >> out_figure10.txt; \
	done
	cat out_figure10.txt


out_figure_9:
	./run.sh; cat out_figure9.txt;
	
figure_9:
	$(MAKE) run_all;
	$(MAKE) out_figure_9;
