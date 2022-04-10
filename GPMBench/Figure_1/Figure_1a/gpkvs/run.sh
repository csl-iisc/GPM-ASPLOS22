file=$1
echo $0 $1
echo "Deleting any older KVS in /pmem folder."
rm -f /pmem/IMKV_insert
rm -f /pmem/imkv.out
echo "Executing gpKVS."
./bin/$file /pmem/gpkvs;
