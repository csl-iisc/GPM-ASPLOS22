#!/bin/bash
APPS=("BFS" "SRAD" "PS")
TYPES=("_gpm", "_mm_gpu", "_fs_gpu")

echo "" >> out_figure9.txt; \
echo -e "\tCAPFs Time\tCAPMM time\tCAPMM speedup\tGPM time\tGPM speedup" >> out_figure9.txt; \
for app in ${APPS[@]}; do \
    capfs_time=$(grep "runtime" results/${app}_fs_gpu.txt | grep -oE "[0-9]+\.[0-9]+"); \
    capmm_time=$(grep "runtime" results/${app}_mm_gpu.txt | grep -oE "[0-9]+\.[0-9]+"); \
    gpm_time=$(grep "runtime" results/${app}_gpm.txt | grep -oE "[0-9]+\.[0-9]+"); \
    capmm_speedup=$(echo "scale=3; $capfs_time/$capmm_time" | bc); \
    gpm_speedup=$(echo "scale=3; $capfs_time/$gpm_time" | bc); \
    if [[ "$app" == "SRAD" ]]; then \
        gpufs_time=$(grep "runtime" results/${app}_gpufs.txt | grep -oE "[0-9]+\.[0-9]+"); \
        gpufs_speedup=$(echo "scale=3; ${capfs_time}/${gpufs_time}" | bc); \
    echo -e "${app}\t${capfs_time}\t${capmm_time}\t${capmm_speedup}\t${gpm_time}\t${gpm_speedup}\t${gpufs_time}\t${gpufs_speedup}" >> out_figure9.txt; \
    else \
    echo -e "${app}\t${capfs_time}\t${capmm_time}\t${capmm_speedup}\t${gpm_time}\t${gpm_speedup}" >> out_figure9.txt; \
    fi \
done \

