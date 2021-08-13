import os
import numpy as np 
import cv2 
import click

@click.command()
@click.option('--data_directory', type = str,
              required=True,
              help="Specify the data directory.")
@click.option('--input_folder', type = str,
              required=True,
              help="Specify the input directory name.")

@click.option('--mask_folder', type = str,
              required=True,
              help="Specify the mask directory name.")


def largest_image_dimensions(data_directory,input_folder,mask_folder):
    cwd = os.getcwd()
    input_path = cwd + "/" + str(data_directory) + "/" + str(input_folder)
    mask_path = cwd + "/" + str(data_directory) + "/" + str(mask_folder)

    max_height = []
    max_width = []

    for file in os.listdir(input_path):
        if file.endswith(".png"):
            img = cv2.imread(input_path + '/' + file, 0)
            img_height, img_width = img.shape
            max_height.append(img_height)
            max_width.append(img_width)
        else:
            continue

    max_height2 = []
    max_width2 = []


    for file in os.listdir(mask_path):
        if file.endswith(".png"):
            mask = cv2.imread(input_path + '/' + file, 0)
            mask_height, mask_width = mask.shape
            max_height2.append(mask_height)
            max_width2.append(mask_width)
        else:
            continue
    

    print("Number of inputs:",len(max_height),"check again",len(max_width))
    print("Number of masks:",len(max_height2),"check again", len(max_width2))
    

    count_height_input = 0
    for i in max_height:
        if i == 121:
            count_height_input += 1
        else:
            continue

    count_height_mask = 0
    for j in max_height2:
        if j == 121:
            count_height_mask += 1
        else:
            continue
    
    count_width_input = 0
    for i in max_width:
        if i <= 480:
            count_width_input += 1
        else:
            continue
    
    count_width_mask = 0
    for j in max_width2:
        if j <= 480:
            count_width_mask += 1
        else:
            continue
    
    print("Number of inputs with 121 pixels height:",count_height_input)
    print("Number of masks with 121 pixels height:", count_height_mask)
    print("Number of inputs <= 480 pixels width is:", count_width_input)
    print("Number of masks <= 480 pixels width is:", count_width_mask)

    print(max_width)
    print('\n')
    print(max_width2)
    print('\n')
    print(max_width == max_width2)

    '''
    max_input_height = max(max_height) 
    max_input_width = max(max_width)

    max_mask_height = max(max_height2)
    max_mask_width = max(max_width2)

    print("Max Input height:", max_input_height)
    print("Max Input width:", max_input_width)
    print("Max Mask height:", max_mask_height)
    print("Max Mask width:", max_mask_width)
    '''
    '''
    max_height.sort(reverse=True)
    max_width.sort(reverse=True)
    max_height2.sort(reverse = True)
    max_width2.sort(reverse=True)

    print(max_height[:])
    print(max_width[:])
    print(max_height2[:])
    print(max_width2[:])

    print(min(max_height))
    print(min(max_width))
    '''

    print(max_height[:])
    print(max_width[:])
    print(max_height2[:])
    print(max_width2[:])

if __name__ == "__main__":
    largest_image_dimensions()



