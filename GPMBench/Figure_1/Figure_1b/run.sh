#!/bin/bash
APPS=("bfs" "srad" "ps")
RUNS=1

echo "" > out_figure1b.txt;
echo -e "\tTime_gpm\tTime_cpu\tSpeedup" >> out_figure1b.txt;
for i in ${RUNS}; do \
    for app in ${APPS[@]}; do \
        time_gpm=$(grep "runtime:" results/${i}_${app}_gpm.raw | grep -oE "[0-9]+\.[0-9]+"); \
        time_memcpy=$(grep "memcpy_time:" results/${i}_${app}_gpm.raw | grep -oE "[0-9]+\.[0-9]+"); \
        time_cpu=$(grep "runtime:" results/${i}_${app}_cpu.raw | grep -oE "[0-9]+\.[0-9]+"); \
        total_time=$(echo "$time_gpm+$time_memcpy" | bc); \
        speedup=$(echo "scale=3; $time_cpu/$total_time" | bc); \
        echo -e "${app}\t${time_gpm}\t${time_cpu}\t${speedup}" >> out_figure1b.txt;
    done \
done

