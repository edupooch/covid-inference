  
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import BatchSampler, RandomSampler

from torch.utils.data import Dataset, DataLoader
from PIL import Image

from architecture.densenet121 import load_densenet121

import time
import os
import cv2
from PIL import Image
import pydicom as dicom

from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import pandas as pd

MODEL_PATH = 'model/densenet_covid.pt'
            
def main():
    parser = argparse.ArgumentParser(description='COVID Inference args')
    parser.add_argument('--dcm-path', type=str, required=True,
                        help='Path of the dicom file to load') 
                                       
    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    val_transform = get_transform()
    
    model = load_densenet121()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)  
    model.eval()
    
    dcm_path = args.dcm_path           
    ds = dicom.dcmread(dcm_path, force=True)
    pixel_array = ds.pixel_array
    image = Image.fromarray(pixel_array) 
    image = image.convert('RGB')
    image = val_transform(image)
    data = image.unsqueeze(0)
    data = data.to(device)
    
    with torch.no_grad():
        output = model(data)
    score = f'{output[0,1].item():.6f}'
    print(score)
    return score

def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

if __name__ == '__main__':
    main()