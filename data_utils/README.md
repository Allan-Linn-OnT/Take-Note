# Data Utils
The majority of data utils tools are intended for use to aid the baseline models as well as retrieve the required training data for the model. Please ensure you run step 1 and step 2 if you are interested in training a baseline model.

## 1. Get mid files
```
bash get_data.sh
```
This will attempt to collect the midi data to train the model and prepare a data folder within the root of the repository. It will also clean the midi files so they can be opened within the note-seq library.

## 2. Convert .mid files to images
```
bash midi_to_pianoroll_image.sh
```
This command will convert the midi files found within the dataset folder of data into images which can be used for training.

A very small number of midi files will fail to convert due to melody inference issues. This is a low priority issue.

## Convert images to .mid files
When you are ready to convert the model output back to a midi for audio evaluation, use the following commands.

single image
```
python midi_to_pianoroll_image.py -d debug_inp.png -o swag.mid
```

batch
```
python midi_to_pianoroll_image.py -i path_to_folder -o output_folder
```
