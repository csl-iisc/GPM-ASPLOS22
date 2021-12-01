mkdir -p results
rm -f results/*
make > ../compile_log.txt

i="10"
echo "Executing GPM with ${i} cp iters..."
./build/lenet_${i}_gpm > ./results/gpm_${i}.txt
echo "Executing GPM-WDP with ${i} cp iters..."
./build/lenet_${i}_gpm_wdp > ./results/gpm_wdp_${i}.txt
echo "Executing Coarse-mm with ${i} cp iters..."
PMEM_THREADS=32 ./build/lenet_${i}_mm_gpu > ./results/coarse_mm_${i}.txt
echo "Executing Coarse-fs with ${i} cp iters..."
./build/lenet_${i}_fs_gpu > ./results/coarse_fs_${i}.txt
echo "Done!"
