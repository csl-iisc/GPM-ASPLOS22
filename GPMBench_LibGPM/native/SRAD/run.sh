mkdir -p results
rm -f results/*
make > ./compile_log.txt

echo "Executing FS..."
./build/srad_fs_gpu 131072 1024 0 31 0 31 0.5 50 > ./results/fs.txt
echo "Executing MM..."
./build/srad_mm_gpu 131072 1024 0 31 0 31 0.5 50 > ./results/mm.txt
echo "Executing GPM..."
./build/srad_gpm 131072 1024 0 31 0 31 0.5 50 > ./results/gpm.txt
echo "Executing GPM-WDP..."
./build/srad_gpm_wdp 131072 1024 0 31 0 31 0.5 50 > ./results/gpm_wdp.txt
echo "Done!"
