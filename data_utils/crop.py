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


def crop_data(data_directory,input_folder,mask_folder):
    cwd = os.getcwd()
    input_path = cwd + "/" + str(data_directory) + "/" + str(input_folder)
    mask_path = cwd + "/" + str(data_directory) + "/" + str(mask_folder)

    cropped_input_path = cwd + "/" + str(data_directory) + "/cropped_" + str(input_folder)
    cropped_mask_path = cwd + "/" + str(data_directory) + "/cropped_" + str(mask_folder)

    os.mkdir(cropped_input_path)
    os.mkdir(cropped_mask_path)

    for file in os.listdir(input_path):
        if file.endswith(".png"):
            img = cv2.imread(input_path + '/' + file, 0)
            new_img = crop_image(img)
            cv2.imwrite(cropped_input_path + "/" + file, new_img)
        else:
            continue
    
    for file in os.listdir(mask_path):
        if file.endswith(".png"):
            img = cv2.imread(mask_path + '/' + file, 0)
            new_img = crop_image(img)
            cv2.imwrite(cropped_mask_path + "/" + file, new_img)
        else:
            continue



def crop_image(img, desired_height = 121, desired_width = 480):
    '''
    img: image in form of img = cv2.imread('ex.jpg')
    desired_height: output height wanted (pixels)
    desired_width: output width wanted (pixels)
    '''
    print(img.shape)
    img_height, img_width = img.shape

    if img_height < desired_height:
        desired_height = img_height
    
    if img_width < desired_width:
        desired_width = img_width
    
    horizontal_remove = (img_width - desired_width) // 2
    vertical_remove = (img_height - desired_height) // 2

    # [y1:(y2 + 1), x1:(x2 + 1)] -> pair (x1, y1) specifying the coordinates of the top left corner and (x2, y2) specifying the coordinates of the bottom right corner of the pixel array
    #()
    crop_img = img[vertical_remove: (img_height-vertical_remove), horizontal_remove: (img_width - horizontal_remove)]

    crop_height, crop_width = crop_img.shape
    if crop_width == desired_width + 1:
        crop_img = crop_img[:,1:]
    
    if crop_height == desired_height + 1:
        crop_img = crop_img[1: ,:]

    return crop_img

if __name__ == "__main__":
    crop_data()
 