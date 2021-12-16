#!/bin/bash

APPS=("DNN" "CFD" "BLK" "HS")

for app in ${APPS[@]}; do
    echo -ne "${app}\t" >> out_figure12.txt;
    awk -v max=0 '{if($1>max){want=$1; max=$1}}END{print want} ' ${app}/bw_${app}.dat >> out_figure12.txt
done
