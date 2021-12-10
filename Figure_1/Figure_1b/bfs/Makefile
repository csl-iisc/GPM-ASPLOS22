BENCHMARKS   ?= bfs_cpu #bfs_gpm 
RUNS         ?= 1


.SILENT: run

run:
	mkdir -p ../results/
	./process_input.sh
	for runs in `seq 1 1 ${RUNS}`; do \
		for file in ${BENCHMARKS}; do \
			echo "" > ../results/$${runs}_$${file}.raw; \
			./run.sh $$file  >> ../results/$${runs}_$${file}.raw 2>&1; \
		done \
	done
