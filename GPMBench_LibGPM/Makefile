folders ?= transactional checkpointing native

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

out_figure_9:
	touch out_figure9.txt; \
	for i in ${folders}; do \
		cd $$i; \
		$(MAKE) out_figure_9; \
		cat out_figure9.txt >> ../out_figure9.txt; \
		cd ..; \
	done
	
out_table_5:
	for i in ${folders}; do \
		cd $$i; \
		$(MAKE) out_table_5; \
		cd ..; \
	done
