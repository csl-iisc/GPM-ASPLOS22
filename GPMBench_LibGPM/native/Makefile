apps ?= SRAD PS BFS 

run_all:
	for i in ${apps}; do \
		cd $$i; \
		make run_all; \
		cd ..; \
	done
	
run_fig10:
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
		time_gpm_ndp=$$(grep "runtime" results/$${i}_gpm.txt | grep -oE "[0-9]+\.[0-9]+");     \
		echo "$$i\t$${time_gpm_ndp}" >> out_figure10.txt; \
	done



out_figure_9:
	./run.sh; cat out_figure9.txt;
	
figure_9:
	$(MAKE) run_all;
	$(MAKE) out_figure_9;
