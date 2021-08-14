#!/usr/bin/env bash

IFS=$'\n'
for d in $( find ../data -maxdepth 2 -type d)
do
    dir=$( basename "$d" )
    if [ $dir == "test" ]  ||  [ $dir == "train" ]  || [ $dir == "valid" ];
    then
        echo $d
        python midi_to_pianoroll_image.py -m $d -o "$d"_image
    fi
done
unset IFS
