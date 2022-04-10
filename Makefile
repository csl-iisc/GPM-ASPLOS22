folder ?= GPMBench

.PHONY: all figure_1 figure_9 table_5 figure_10 figure_11a out_figure_1 out_figure_9 out_table_5 out_figure_10 out_figure_11a

all:
	cd ${folder}; $(MAKE) all
	
figure_1:
	cd ${folder}; $(MAKE) figure_1

figure_9:
	cd ${folder}; $(MAKE) figure_9;

figure_10:
	cd ${folder}; $(MAKE) figure_9; # Have to get results for Fig 9 first. Used for Fig 10.
	cd ${folder}; $(MAKE) figure_10;
	
figure_11a:
	cd ${folder}; $(MAKE) figure_11a;
	
table_5:
	cd ${folder}; $(MAKE) table_5;

out_figure_1:
	mkdir -p reports/
	cd ${folder}; $(MAKE) out_figure_1;
	cp ${folder}/out_figure1a.txt reports/
	cp ${folder}/out_figure1b.txt reports/

out_figure9:
	mkdir -p reports/
	cd ${folder}; $(MAKE) out_figure_9;
	cp ${folder}/out_figure9.txt reports/;

out_figure10:
	mkdir -p reports/
	cd ${folder}; $(MAKE) out_figure_10;
	cp ${folder}/out_figure10.txt reports/;

out_figure11a:
	mkdir -p reports/
	cd ${folder}; $(MAKE) out_figure_11a;
	cp ${folder}/out_figure11a.txt reports/;
	
out_table_5:
	mkdir -p reports/
	cd ${folder}; $(MAKE) out_table_5;
	cp ${folder}/out_table5.txt reports/;
	
