apps ?= gpKVS gpDB
run_all:
	mkdir -p ./results;
	for i in ${apps}; do \
		cd $$i; \
		make run_all; \
		cd ..; \
	done

run_rest:
	mkdir -p ./results;
	for i in ${apps}; do \
		cd $$i; \
		make run_rest; \
		cd ..; \
	done

run_fig10:
	mkdir -p ./results;
	for i in ${apps}; do \
		cd $$i; \
		make run_fig_10; \
		cd ..; \
	done

out_figure_9:
	echo "" > out_figure9.txt; 
	
	echo "\tCAP-FS\tCAP-MM\tCAPMM Speedup\tGPM\tGPM Speedup" >> out_figure9.txt;
	time_gpm=$$(grep "Runtime" results/kvs_gpm.txt | grep -oE "[0-9]+\.[0-9]+");     \
	time_fs=$$(grep "Runtime" results/kvs_fs_gpu.txt | grep -oE "[0-9]+\.[0-9]+");   \
	time_mm=$$(grep "Runtime" results/kvs_mm_gpu.txt | grep -oE "[0-9]+\.[0-9]+");   \
	speedup_mm=$$(awk "BEGIN {print $${time_fs}/$${time_mm}}"); \
	speedup_gpm=$$(awk "BEGIN {print $${time_fs}/$${time_gpm}}"); \
	echo "gpKVS\t$${time_fs}\t$${time_mm}\t$${speedup_mm}\t$${time_gpm}\t$${speedup_gpm}" >> out_figure9.txt;
	
	#echo "\tCAP-FS\tCAP-MM\tGPM" >> out_figure9.txt;
	time_gpm=$$(grep "Runtime" results/kvs_gs_gpm.txt | grep -oE "[0-9]+\.[0-9]+");     \
	time_fs=$$(grep "Runtime" results/kvs_gs_fs_gpu.txt | grep -oE "[0-9]+\.[0-9]+");   \
	time_mm=$$(grep "Runtime" results/kvs_gs_mm_gpu.txt | grep -oE "[0-9]+\.[0-9]+");   \
	speedup_mm=$$(awk "BEGIN {print $${time_fs}/$${time_mm}}"); \
	speedup_gpm=$$(awk "BEGIN {print $${time_fs}/$${time_gpm}}"); \
	echo "gpKVS_GS\t$${time_fs}\t$${time_mm}\t$${speedup_mm}\t$${time_gpm}\t$${speedup_gpm}" >> out_figure9.txt;
	
	#echo "\tCAP-FS\tCAP-MM\tGPM" >> out_figure9.txt;
	time_gpm=$$(grep "InsertTime" results/gpDB_gpm.txt | grep -oE "[0-9]+\.[0-9]+");     \
	time_fs=$$(grep "InsertTime" results/gpDB_fs_gpu.txt | grep -oE "[0-9]+\.[0-9]+");   \
	time_mm=$$(grep "InsertTime" results/gpDB_mm_gpu.txt | grep -oE "[0-9]+\.[0-9]+");   \
	speedup_mm=$$(awk "BEGIN {print $${time_fs}/$${time_mm}}"); \
	speedup_gpm=$$(awk "BEGIN {print $${time_fs}/$${time_gpm}}"); \
	echo "gpDB_I\t$${time_fs}\t$${time_mm}\t$${speedup_mm}\t$${time_gpm}\t$${speedup_gpm}" >> out_figure9.txt;
	
	#echo "\tCAP-FS\tCAP-MM\tGPM" >> out_figure9.txt;
	time_gpm=$$(grep "UpdateTime" results/gpDB_gpm.txt | grep -oE "[0-9]+\.[0-9]+");     \
	time_fs=$$(grep "UpdateTime" results/gpDB_fs_gpu.txt | grep -oE "[0-9]+\.[0-9]+");   \
	time_mm=$$(grep "UpdateTime" results/gpDB_mm_gpu.txt | grep -oE "[0-9]+\.[0-9]+");   \
	speedup_mm=$$(awk "BEGIN {print $${time_fs}/$${time_mm}}"); \
	speedup_gpm=$$(awk "BEGIN {print $${time_fs}/$${time_gpm}}"); \
	echo "gpDB_U\t$${time_fs}\t$${time_mm}\t$${speedup_mm}\t$${time_gpm}\t$${speedup_gpm}" >> out_figure9.txt;
	
	cat out_figure9.txt;
	
out_table_5:
	echo "" > out_table5.txt;
	
	time_gpm=$$(grep "Runtime" results/kvs_gpm.txt | grep -oE "[0-9]+\.[0-9]+");     \
	time_recov=$$(grep "Recovery" results/kvs_gpm_rest.txt | grep -oE "[0-9]+\.[0-9]+");     \
	time=$$(awk "BEGIN {print $${time_recov}/$${time_gpm}}"); \
	echo "KVS\t$${time}" >> out_table5.txt;
	
	time_gpm=$$(grep "InsertTime" results/gpDB_gpm.txt | grep -oE "[0-9]+\.[0-9]+");     \
	time_recov=$$(grep "InsertRecovery" results/gpDB_gpm_rest.txt | grep -oE "[0-9]+\.[0-9]+");     \
	time=$$(awk "BEGIN {print $${time_recov}/$${time_gpm}}"); \
	echo "gpDB_I\t$${time}" >> out_table5.txt;
	
	time_gpm=$$(grep "UpdateTime" results/gpDB_gpm.txt | grep -oE "[0-9]+\.[0-9]+");     \
	time_recov=$$(grep "UpdateRecovery" results/gpDB_gpm_rest.txt | grep -oE "[0-9]+\.[0-9]+");     \
	time=$$(awk "BEGIN {print $${time_recov}/$${time_gpm}}"); \
	echo "gpDB_U\t$${time}" >> out_table5.txt;
	
	cat out_table5.txt;
	
ndp_apps=kvs kvs_gs

out_figure_10:
	echo "" > out_figure10.txt;
	for i in ${ndp_apps}; do                                            \
		time_gpm_ndp=$$(grep "Runtime" results/$${i}_gpm_ndp.txt | grep -oE "[0-9]+\.[0-9]+");     \
		time_gpm_ndp_persist=$$(grep "PersistTime" results/$${i}_gpm_ndp.txt | grep -oE "[0-9]+\.[0-9]+");     \
		time_capfs=$$(grep "Runtime" results/$${i}_fs_gpu.txt | grep -oE "[0-9]+\.[0-9]+");     \
		time_capmm_persist=$$(grep "PersistTime" results/$${i}_mm_gpu.txt | grep -oE "[0-9]+\.[0-9]+");     \
		speedup_gpm_ndp=$$(awk "BEGIN {print $${time_capfs}/$${time_gpm_ndp}}"); \
		echo "$$i\t$${time_gpm_ndp}\t$${time_gpm_ndp_persist}\t$${speedup_gpm_ndp}\t$${time_capmm_persist}" >> out_figure10.txt; \
	done
	cat out_figure10.txt



figure_9:
	$(MAKE) run_all TYPES="_gpm _fs_gpu _mm_gpu";
	$(MAKE) out_figure_9;

figure_10:
	$(MAKE) run_fig10;
	$(MAKE) out_figure_10;
	
table_5:
	$(MAKE) run_rest TYPES="_gpm _gpm_rest";
	$(MAKE) out_table_5;	
