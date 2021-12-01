#!/bin/bash

file=$1
ps_cpu_file="ps_cpu"
echo $0 $1
if [[ "$file" == "$ps_cpu_file" ]]; then
    ./bin/$file 32;
else
    ./bin/$file ;
fi

