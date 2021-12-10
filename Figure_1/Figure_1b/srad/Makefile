BENCHMARKS   ?= srad_cpu srad_gpm 
RUNS         ?= 1


.SILENT: run

run:
	mkdir -p ../results/
	for runs in `seq 1 1 ${RUNS}`; do \
		for file in ${BENCHMARKS}; do \
			echo "" > ../results/$${runs}_$${file}.raw; \
			./run.sh $$file  >> ../results/$${runs}_$${file}.raw 2>&1; \
		done \
	done
