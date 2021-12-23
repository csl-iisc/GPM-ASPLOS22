folder ?= GPMBench_LibGPM

figure_1:
	cd Figure_1; $(MAKE) run_all;

figure_9:
	cd ${folder}; $(MAKE) figure_9;

figure_10:
	cd ${folder}; $(MAKE) figure_9; # Have to get results for Fig 9 first. Used for Fig 10.
	cd ${folder}; $(MAKE) figure_10;
	
figure_11a:
	cd ${folder}; $(MAKE) figure_11a;
	
table_5:
	cd ${folder}; $(MAKE) table_5;

all:
	make figure_1 
	make figure_9 
	make table_5 
	make figure_10
	make figure_11a

out_figure1:
	mkdir -p reports/
	cd Figure_1; $(MAKE) out_figure1;
	mv Figure_1/out_figure1a.txt reports/ 
	mv Figure_1/out_figure1b.txt reports/ 
	cat reports/out_figure1a.txt	
	cat reports/out_figure1b.txt	

out_figure9:
	mkdir -p reports/
	cd ${folder}; $(MAKE) out_figure_9;
	mv GPMBench_LibGPM/out_figure9.txt reports/;
	cat reports/out_figure9.txt

out_figure10:
	mkdir -p reports/
	cd ${folder}; $(MAKE) out_figure_10;
	mv GPMBench_LibGPM/out_figure10.txt reports/;
	cat reports/out_figure10.txt

out_figure11a:
	mkdir -p reports/
	cd ${folder}; $(MAKE) out_figure_11a;
	mv GPMBench_LibGPM/out_figure11a.txt reports/;
	cat reports/out_figure11a.txt
	
out_table_5:
	mkdir -p reports/
	cd ${folder}; $(MAKE) out_table_5;
	mv GPMBench_LibGPM/out_table5.txt reports/;
	cat reports/out_table5.txt
	
