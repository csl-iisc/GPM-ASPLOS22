folders ?= transactional checkpointing native
rest_folders ?= transactional checkpointing 

figure_9:
	for i in ${folders}; do \
		cd $$i; \
		$(MAKE) figure_9; \
		cd ..; \
	done
	
table_5:
	for i in ${folders}; do \
		cd $$i; \
		$(MAKE) table_5; \
		cd ..; \
	done

figure_10:
	for i in ${folders}; do \
		cd $$i; \
		$(MAKE) figure_10; \
		cd ..; \
	done

figure_11a:
	cd transactional; \
	$(MAKE) figure_11a;

out_figure_9:
	touch out_figure9.txt; \
	for i in ${folders}; do \
		cd $$i; \
		$(MAKE) out_figure_9; \
		cat out_figure9.txt >> ../out_figure9.txt; \
		cd ..; \
	done
	
out_table_5:
	touch out_table5.txt; \
	for i in ${rest_folders}; do \
		cd $$i; \
		$(MAKE) out_table_5; \
		cat out_table5.txt >> ../out_table5.txt; \
		cd ..; \
	done

out_figure_10:
	touch out_figure10.txt; \
	for i in ${folders}; do \
		cd $$i; \
		$(MAKE) out_figure_10; \
		cat out_figure10.txt >> ../out_figure10.txt; \
		cd ..; \
	done

out_figure_11a:
	touch out_figure11a.txt; \
	cd transactional; \
	$(MAKE) out_figure_11a; \
	cat out_figure11a.txt >> ../out_figure11a.txt;
