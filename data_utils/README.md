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

# Standardize Image Dimensions by first cropping and then padding

```
python3 crop.py --data_directory <data_folder_name> --input_folder <input_folder_name> --mask_folder <mask_folder_name>
```

Data Folder Hierarchy:

<data_folder_name>  contains  <input_folder_name>  and  <mask_folder_name> 

```
python3 pad_images_f.py --data_directory <data_folder_name> --input_folder <input_folder_name> --mask_folder <mask_folder_name>
```

# To Print Image Dimensions in Input Folder and Mask Folder
```
python3 find_dimensions.py --data_directory <data_folder_name> --input_folder <input_folder_name> --mask_folder <mask_folder_name>
```
```
