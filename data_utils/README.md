# Get mid files
```
bash get_data.sh
```
a very small number of midi files will fail to convert due to melody inference issues. This is a low priority issue.

# Convert .mid files to images
```
bash midi_to_pianoroll_image.sh
```

# Convert images to .mid files
single image
```
python midi_to_pianoroll_image.py -d debug_inp.png -o swag.mid
```

batch
```
python midi_to_pianoroll_image.py -i path_to_folder -o output_folder
```
```
