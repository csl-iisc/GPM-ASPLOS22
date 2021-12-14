#!/bin/bash

file=$1
srad_cpu_file="srad_cpu"
echo $0 $1
if [[ "$file" == "$srad_cpu_file" ]]; then
    rm -f /pmem/persist_e_c.dat
    rm -f /pmem/persist_n_c.dat
    rm -f /pmem/persist_s_c.dat
    rm -f /pmem/persist_w_c.dat
    rm -f /pmem/persist_j.dat
    rm -f /pmem/persist_j_out.dat
    rm -f /pmem/persist_c.dat
    ./bin/$file 131072 1024 0 31 0 31 0.5 50 32;
else
    ./bin/$file 131072 1024 0 31 0 31 0.5 50;
fi

