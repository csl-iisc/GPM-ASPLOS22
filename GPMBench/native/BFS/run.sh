mkdir -p results
rm -f results/*
make > ./compile_log.txt

echo "Executing FS..."
./build/bfs_fs > ./results/fs.txt
echo "Executing MM..."
./build/bfs_gpu > ./results/mm.txt
echo "Executing GPM-far..."
./build/bfs_gpm > ./results/gpm.txt
echo "Done!"
