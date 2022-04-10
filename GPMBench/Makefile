folders ?= transactional checkpointing native
rest_folders ?= transactional checkpointing 

.PHONY: all figure_1 figure_9 table_5 figure_10 figure_11a out_figure_1 out_figure_9 out_table_5 out_figure_10 out_figure_11a

all:
	$(MAKE) figure_1 
	$(MAKE) figure_9 
	$(MAKE) figure_10
	$(MAKE) figure_11a
	$(MAKE) table_5 
	
figure_1:
	cd Figure_1; $(MAKE) run_all;
	$(MAKE) out_figure_1;

figure_9:
	for i in ${folders}; do \
		cd $$i; \
		$(MAKE) figure_9; \
		cd ..; \
	done
	$(MAKE) out_figure_9;
	
table_5:
	for i in ${folders}; do \
		cd $$i; \
		$(MAKE) table_5; \
		cd ..; \
	done
	$(MAKE) out_table_5;

figure_10:
	for i in ${folders}; do \
		cd $$i; \
		$(MAKE) figure_10; \
		cd ..; \
	done
	$(MAKE) out_figure_10;

figure_11a:
	cd transactional; \
	$(MAKE) figure_11a;
	$(MAKE) out_figure_11a;


out_figure_1:
	cd Figure_1; $(MAKE) out_figure1;
	mv Figure_1/out_figure1a.txt
	mv Figure_1/out_figure1b.txt
	cat reports/out_figure1a.txt	
	cat reports/out_figure1b.txt	

out_figure_9:
	touch out_figure9.txt; \
	for i in ${folders}; do \
		cd $$i; \
		$(MAKE) out_figure_9; \
		cat out_figure9.txt >> ../out_figure9.txt; \
		cd ..; \
	done
	cat out_figure9.txt
	
out_table_5:
	touch out_table5.txt; \
	for i in ${rest_folders}; do \
		cd $$i; \
		$(MAKE) out_table_5; \
		cat out_table5.txt >> ../out_table5.txt; \
		cd ..; \
	done
	cat out_table5.txt

out_figure_10:
	touch out_figure10.txt; \
	for i in ${folders}; do \
		cd $$i; \
		$(MAKE) out_figure_10; \
		cat out_figure10.txt >> ../out_figure10.txt; \
		cd ..; \
	done
	cat out_figure10.txt

out_figure_11a:
	touch out_figure11a.txt; \
	cd transactional; \
	$(MAKE) out_figure_11a; \
	cat out_figure11a.txt >> ../out_figure11a.txt;
	cat out_figure11a.txt
