mkdir -p results
rm -f results/*
make > ./compile_log.txt

echo "Executing volatile..."
./build/kvs_volatile > ./results/volatile.txt
echo "Executing GPM..."
./build/kvs_gpm > ./results/gpm_far.txt
echo "Executing GPM-get-set..."
./build/kvs_get_set_gpm > ./results/gpm_get_set.txt
echo "Executing GPM-ndp..."
./build/kvs_gpm_wdp > ./results/gpm_wdp.txt
echo "Executing CAP-fs..."
./build/kvs_fs_gpu > ./results/coarse_fs.txt
echo "Executing CAP-mm..."
./build/kvs_mm_gpu > ./results/coarse_mm.txt
#echo "Executing Coarse-mm-tx..."
#./build/imkv_mm_tx_gpu > ./results/coarse_mm_tx.txt
#echo "Executing Coarse-tx..."
#%./imkv_tx_gpu > ./results/coarse_tx.txt
echo "Done!"
