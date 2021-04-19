from __future__ import print_function
import argparse
import torch
from torchvision import transforms
from .architecture.densenet121 import load_densenet121
from PIL import Image
import pydicom as dicom

from .interpretability.grad_cam import GradCAM, GradCamPlusPlus
from skimage import io
from torch import nn
import numpy  as np
import cv2

MODEL_PATH = './covid_inference/model/densenet_covid.pt'
# MODEL_PATH = 'model/densenet_covid_fulldata.pt'

def gen_cam(image, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    cam = heatmap + np.float32(image)
    return norm_image(cam), (heatmap * 255).astype(np.uint8)

def norm_image(image):
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def prepare_input(image):
    image = image.copy()

    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1))) 
    image = image[np.newaxis, ...] 

    return torch.tensor(image, requires_grad=True)

def inference(file):    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    val_transform = get_transform()
    
    model = load_densenet121()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)  
    model.eval()
    net = model
    
    dcm_path = file          
    ds = dicom.dcmread(dcm_path, force=True)
    pixel_array = ds.pixel_array
    image = Image.fromarray(pixel_array) 
    image = image.convert('RGB')
    data = val_transform(image)
    data = data.unsqueeze(0)
    data = data.to(device)

    # Grad-CAM++
    img = np.array(image)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    inputs = data
    inputs = prepare_input(img)
    inputs = inputs.to(device)
    layer_name = "features.denseblock4.denselayer16.conv2"
    grad_cam_plus_plus = GradCamPlusPlus(net, layer_name)
    mask_plus_plus = grad_cam_plus_plus(inputs, None)  # cam mask
    cam, heatmap = gen_cam(img, mask_plus_plus)
    grad_cam_plus_plus.remove_handlers()
    # io.imsave('cam.jpg', cam)
    
    with torch.no_grad():
        output = model(data)
    score = f'{output[0,1].item():.6f}'
    
    return score, cam

def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])