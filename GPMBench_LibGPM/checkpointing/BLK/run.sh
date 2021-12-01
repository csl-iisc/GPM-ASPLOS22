mkdir -p results
rm -f results/*

#iters=("5" "10")
i="10"

make all CP_ITER=$i > ./compile_log.txt
echo "Executing GPM with ${i} cp iters..."
./build/blackScholes_${i}_gpm > ./results/gpm_${i}.txt
echo "Executing GPM-WDP with ${i} cp iters..."
./build/blackScholes_${i}_gpm_wdp > ./results/gpm_wdp_${i}.txt
echo "Executing Coarse-fs with ${i} cp iters..."
./build/blackScholes_${i}_fs_gpu > ./results/coarse_fs_${i}.txt
echo "Executing Coarse-mm with ${i} cp iters..."
./build/blackScholes_${i}_mm_gpu > ./results/coarse_mm_${i}.txt
echo "Done!"
