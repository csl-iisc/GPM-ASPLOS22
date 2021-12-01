mkdir -p results
rm -f results/*
make > ./compile_log.txt

#echo "Executing volatile..."
#./build/shfl_scan_gpu > ./results/volatile.txt
echo "Executing FS..."
./build/shfl_scan_fs_gpu > ./results/fs.txt
echo "Executing MM..."
./build/shfl_scan_mm_gpu > ./results/mm.txt
echo "Executing GPM-far..."
./build/shfl_scan_real_cpu > ./results/gpm_far.txt
#echo "Executing GPM-near..."
#./build/shfl_scan_emul_gpu > ./results/gpm_near.txt
echo "Done!"
