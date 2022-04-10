file=$1
echo $0 $1
echo "Deleting any older KVS in /pmem folder."
rm -f /pmem/rocksdb_kvs/*
mkdir -p /pmem/rocksdb_value
rm -f /pmem/rocksdb_kvs
echo "Executing RocksDB."
./bin/$file;
