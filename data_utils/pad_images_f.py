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

def pad_data(data_directory,input_folder,mask_folder):
    '''
    This function pads all the png images in the input and mask folders
    in a given data_directory and returns padded images in new folders.

    The hirechy od the parameters should be:

    data_directory
    ---------------input_folder
    ---------------mask_folder

    Where the input_folder and mask_folder are within the data_directory folder

    data_directory: Name of data directory folder
    input_folder: Name of input images folder
    mask_folder: Name of mask images folder
    '''

    cwd = os.getcwd()
    input_path = cwd + "/" + str(data_directory) + "/" + str(input_folder)
    mask_path = cwd + "/" + str(data_directory) + "/" + str(mask_folder)
    
    resized_input_path = cwd + "/" + str(data_directory) + "/resized_" + str(input_folder)
    resized_mask_path = cwd + "/" + str(data_directory) + "/resized_" + str(mask_folder)
    os.mkdir(resized_input_path)
    os.mkdir(resized_mask_path)

    for file in os.listdir(input_path):
        if file.endswith(".png"):
            img = cv2.imread(input_path + '/' + file, 0)
            new_img = pad_image(img)
            cv2.imwrite(resized_input_path + "/" + file, new_img)
        else:
            continue

    for file in os.listdir(mask_path):
        if file.endswith(".png"):
            img = cv2.imread(mask_path + '/' + file, 0)
            new_img = pad_image(img)
            cv2.imwrite(resized_mask_path + "/" + file, new_img)
        else:
            continue


def pad_image(img, desired_height=224, desired_width=480, colour = 0):
    '''
    img: image in form of img = cv2.imread('ex.jpg')
    desired_height: output height wanted (pixels)
    desired_width: output width wanted (pixels)
    colour: 0 to 255, default is 0 (black)
    '''
    img_height, img_width = img.shape

    if img_width > desired_width:
        desired_width = img_width
    
    if img_height > desired_height:
        desired_height = img_height

    result = np.full((desired_height,desired_width),colour,dtype=np.uint8)
    xx = (desired_width - img_width) // 2
    yy = (desired_height - img_height) // 2
    result[yy:yy+img_height, xx:xx+img_width] = img
    
    return result


if __name__ == "__main__":
    pad_data()






