import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from PIL import Image

# Turn off Gradient calcuation during inference
torch.no_grad()

def normalize(arr):
    return (arr-np.min(arr))/(np.max(arr)-np.min(arr))

preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Resize(size=(512,512)),
    ])


model = torch.load("NH_Model/weights.pt")
model.eval()
model.to('cpu')
model.requires_grad = False

#img = Image.open('/home/martin/Desktop/RobVol2.5/APS360/Project/TakeNote/data/JSB Chorales/valid_image/input/136.png').convert("RGB")
img = Image.open('/home/martin/Desktop/RobVol2.5/APS360/Project/TakeNote/data/Nottingham/test_image/input/ashover_simple_chords_11.png').convert("RGB")

#img = Image.open('_swag.png').convert("RGB")
nimg = np.asarray(img)[..., 0] # only first channel is needed
h, w = nimg.shape
print(nimg.shape)
outputs = model(torch.unsqueeze(preprocess(img), 0))

nout = normalize(outputs.detach().numpy()[0, 0, :])

threshold = 0.70
plt.imshow(nout>threshold)
plt.show()
nout[nout>threshold] = 1
nout[nout<=threshold] = 0
nout = cv2.resize(nout, (w, h)) + nimg # recover original melody
Image.fromarray(255*nout).convert("RGB").save('swag.png')
