file=$1
echo $0 $1
echo "Deleting any older KVS in /pmem folder."
rm -f /pmem/pmemkv_kvs
echo "Executing Intel PmemKV."
./bin/$file /pmem/pmemkv_kvs;
