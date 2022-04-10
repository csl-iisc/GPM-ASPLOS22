mkdir -p results
rm -f results/*

i="10"
make all CP_ITER=$i > ./compile_log.txt
echo "Executing GPM with ${i} cp iters..."
./build/euler3d_${i}_gpm data/missile.domn.0.2M > ./results/gpm_${i}.txt
echo "Executing GPM-WDP with ${i} cp iters..."
./build/euler3d_${i}_gpm_wdp data/missile.domn.0.2M > ./results/gpm_wdp_${i}.txt
echo "Executing Coarse-mm with ${i} cp iters..."
./build/euler3d_${i}_mm_gpu data/missile.domn.0.2M > ./results/coarse_mm_${i}.txt
echo "Executing Coarse-fs with ${i} cp iters..."
./build/euler3d_${i}_fs_gpu data/missile.domn.0.2M > ./results/coarse_fs_${i}.txt
echo "Done!"
