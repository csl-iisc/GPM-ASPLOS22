#!/bin/bash
CP_ITER=10
APPS=("BLK" "CFD" "DNN" "HS")

echo "" >> out_figure9.txt; \
echo -e "\tCAPFs Time\tCAPMM time\tCAPMM speedup\tGPM time\tGPM speedup\tGPUFs time \tGPUFs speedup" >> out_figure9.txt; \
for app in ${APPS[@]}; do \
    capfs_time=$(grep "CheckpointTime" results/${app}_${CP_ITER}_fs_gpu.txt | grep -oE "[0-9]+\.[0-9]+"); \
    capmm_time=$(grep "CheckpointTime" results/${app}_${CP_ITER}_mm_gpu.txt | grep -oE "[0-9]+\.[0-9]+"); \
    gpm_time=$(grep "CheckpointTime" results/${app}_${CP_ITER}_gpm.txt | grep -oE "[0-9]+\.[0-9]+"); \
    capmm_speedup=$(echo "scale=3; ${capfs_time}/${capmm_time}" | bc); \
    gpm_speedup=$(echo "scale=3; ${capfs_time}/${gpm_time}" | bc); \
    if [[ "$app" == "DNN" || "$app" == "CFD" ]]; then \
        gpufs_time=$(grep "CheckpointTime" results/${app}_${CP_ITER}_gpufs.txt | grep -oE "[0-9]+\.[0-9]+"); \
        gpufs_speedup=$(echo "scale=3; ${capfs_time}/${gpufs_time}" | bc); \
        echo -e  "${app}\t${capfs_time}\t${capmm_time}\t${capmm_speedup}\t${gpm_time}\t${gpm_speedup}\t${gpufs_time}\t${gpufs_speedup}" >> out_figure9.txt; \
    else \
    echo -e "${app}\t${capfs_time}\t${capmm_time}\t${capmm_speedup}\t${gpm_time}\t${gpm_speedup}" >> out_figure9.txt; \
    fi
done

