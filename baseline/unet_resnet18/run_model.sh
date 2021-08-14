#!/usr/bin/env bash
# arguments: 

# 	1 - path to folder filled with the samples
# 	2 - path to desired output folder
# 	3 - path to model weights
#	4 - threshold
#	5 - model name

if [ "$#" -ne 5 ]; then
	echo "Need 5 Arguments, open script for details"
	exit 0
fi

echo "Running model inference on samples"
echo "Samples Folder path: $1"
echo "Output Folder path: $2"
echo "Model Weights path: $3"
echo "Threshold: $4"
echo "Model Name: $5"

for f in $( find $1 -name "*.png");do
	echo $f
	output_file="$(basename $f)"
	python infer.py -s $f -o $2/$output_file -w $3 -t $4
	python ../../data_utils/midi_to_pianoroll_image.py -d $2/$output_file -o $2/$5_"${output_file%.png}.mid"
done
