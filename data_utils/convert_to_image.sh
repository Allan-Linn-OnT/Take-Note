#!/bin/bash

# DEPRECATED, DO NOT USE 
IFS=$'\n'
for d in $( find ../data -maxdepth 2 -type d)
do
    dir=$( basename "$d" )
    if [ $dir == "test" ]  ||  [ $dir == "train" ]  || [ $dir == "valid" ];
    then
        echo $d
        python convert_to_image.py -m $d -o "$d"_input -r 1 -p True
        python convert_to_image.py -m $d -o "$d"_target -r 1  
    fi
done
unset IFS