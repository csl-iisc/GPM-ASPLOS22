#!/bin/bash

file=$1
srad_cpu_file="srad_cpu"
echo $0 $1
if [[ "$file" == "$srad_cpu_file" ]]; then
    ./bin/$file 131072 1024 0 31 0 31 0.5 50 32;
else
    ./bin/$file 131072 1024 0 31 0 31 0.5 50;
fi

