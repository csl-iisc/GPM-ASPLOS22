file=$1
echo $0 $1
echo "Deleting any older KVS in /pmem folder."
rm -f /pmem/matrixkv_kvs
rm -rf /tmp/matrixkv/
rm -f cf_0_default_sstable.pool
rm -f NVM_LOG
rm -f logs/*
echo "Executing MatrixKV."
./bin/$file /pmem/matrixkv_kvs;
