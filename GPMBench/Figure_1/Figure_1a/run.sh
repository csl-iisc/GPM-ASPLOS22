#!/bin/bash
APPS=("rocksdb" "gpkvs" "pmemkv" "matrixkv")
RUNS=1
OPERATIONS=52428800

echo "" > out_figure1a.txt; \
echo -e "\tOperations\tTime\tThroughput" >> out_figure1a.txt; \
for i in ${RUNS}; do \
    for app in ${APPS[@]}; do \
        time=$(grep "runtime:" results/${i}_${app}.raw | grep -oE "[0-9]+\.[0-9]+"); \
        throughput=$(echo "scale=3; $OPERATIONS/$time" | bc); \
        echo -e "${app}\t${OPERATIONS}\t${time}\t${throughput}" >> out_figure1a.txt ; \
    done \
done

