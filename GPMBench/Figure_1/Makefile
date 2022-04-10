run_figure1a: 
	cd Figure_1a/; sudo make run; cd ../

run_figure1b: 
	cd Figure_1b/; sudo make run; cd ../ 

run_all: 
	make run_figure1a
	make run_figure1b

out_figure1a: 
	cd Figure_1a/; \
		make out_figure_1a; \
	cd ../;
	mv Figure_1a/out_figure1a.txt .

out_figure1b: 
	cd Figure_1b/; make out_figure1b; cd ../; mv Figure_1b/out_figure1b.txt .

out_figure1: 
	make out_figure1a
	make out_figure1b
