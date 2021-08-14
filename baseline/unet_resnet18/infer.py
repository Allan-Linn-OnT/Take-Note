import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import pathlib
import glob
import os

from torchvision import transforms
from PIL import Image

# Turn off Gradient calcuation during inference
torch.no_grad()

def element_logistic(arr):
    return 1/( 1 + np.exp(-arr))

def normalize(arr):
    return element_logistic(arr) #(arr-np.min(arr))/(np.max(arr)-np.min(arr))


preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Resize(size=(512,512)),
    ])

def process_file(img_file, output_path, threshold, model, visualize=True):
    img = Image.open(img_file).convert("RGB")
    nimg = np.asarray(img)[..., 0] # only first channel is needed
    h, w = nimg.shape
    outputs = model(torch.unsqueeze(preprocess(img), 0))
    nout = normalize(outputs.detach().numpy()[0, 0, :])
    
    if visualize:
        plt.imshow(nout>threshold)
        plt.show()
    
    nout[nout>threshold] = 1
    nout[nout<=threshold] = 0
    nout = cv2.resize(nout, (w, h)) + nimg # recover original melody
    Image.fromarray(255*nout).convert("RGB").save(output_path)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--model_weights', required=True, help='specify path to the weights')
    parser.add_argument('-o','--output', required=True, help='specify path to save the model outputs to')
    parser.add_argument('-s','--sample', default=None, help='midi sample we want to convert')
    parser.add_argument('-d', '--dir', default=None, help='directory path to which we want to convert')
    parser.add_argument('-t', '--threshold', default=0.6, help='threshold')
    args = parser.parse_args()

    model = torch.load(args.model_weights)
    model.eval()
    model.to('cpu')
    model.requires_grad = False
    
    if args.sample is not None:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        process_file(args.sample, args.output, float(args.threshold), model)
        exit(0)

    if args.dir is not None:
        os.makedirs(args.output, exist_ok=True)
        for img_file in glob.glob(f"{args.dir}/*.png"):
            process_file(img_file, f"{output_path}/{os.path.basename(img_file)}", float(args.threshold), model)
        exit(0)
if __name__ == '__main__':
    main()
