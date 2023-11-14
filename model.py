# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:07:05 2023

@author: DELL
"""
# Importing all packages 
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import torch
from torch import nn
from torch import optim
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image, ImageFile
import json
from torch.optim import lr_scheduler
import random
import os
import sys

print('Imported packages')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet152(pretrained=False)
num_ftrs = model.fc.in_features
out_ftrs = 5
model.fc = nn.Sequential(nn.Linear(num_ftrs, 512),nn.ReLU(),nn.Linear(512,out_ftrs),nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()) , lr = 0.00001)

scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
model.to(device);
# to unfreeze more layers


for name,child in model.named_children():
    if name in ['layer2','layer3','layer4','fc']:
        #print(name + 'is unfrozen')
        for param in child.parameters():
            param.requires_grad = True
    else:
        #print(name + 'is frozen')
        for param in child.parameters():
            param.requires_grad = False
optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()) , lr = 0.000001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

def load_model(path):
    checkpoint = torch.load(path,map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model
def inference(model, file, transform, classes):
    file = Image.open(file).convert('RGB')
    img = transform(file).unsqueeze(0)
    print('Transforming your image...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        print('Passing your image to the model....')
        out = model(img.to(device))
        ps = torch.exp(out)
        top_p, top_class = ps.topk(1, dim=1)
        value = top_class.item()
        print("Predicted Severity Value: ", value)
        print("class is: ", classes[value])
        print('Your image is printed:')
        return value, classes[value]
        # plt.imshow(np.array(file))
        # plt.show()


model = load_model('C:/Users/DELL/OneDrive/Desktop/minor projrct/classifier.pt')
print("Model loaded Succesfully")
classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
def main(path):
    x, y = inference(model, path, test_transforms, classes)
    return x, y
from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
from PIL import Image
import os
from tkinter.filedialog import askopenfilename, asksaveasfilename
def OpenFile():
    try:
        a = askopenfilename()
        print(a)
        value, classes = main(a)
        messagebox.showinfo("your report", ("Predicted Label is ", value, "\nPredicted Class is ", classes))
        image = Image.open(a)
        # plotting image
        file = image.convert('RGB')
        plt.imshow(np.array(file))
        plt.title(f'your report is label : {value} class : {classes}')
        plt.show()
        #print(image)
        print('Thanks for using the system !')
        #fn, text = os.path.splitext(a) #fn stands for filename
    except Exception as error:
        print("File not selected ! Exiting..., Please try again")
root = Tk()
root.geometry('700x400')
root.title("Diabetic Retinopathy Detection System")

root.configure(bg='ivory3')
button = Button(root, text="Upload Image", command=OpenFile)
button.grid(padx=10, pady=20, row=2, column=3)
root.mainloop()