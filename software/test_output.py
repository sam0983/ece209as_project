import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.optim as optim
import socket
import pickle
import json
import numpy as np
from timeit import default_timer as timer
import sys

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

original_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class vgg(nn.Module):
            def __init__(self):
                super(vgg, self).__init__()
                self.features = nn.Sequential(
 		    *(list(original_model.features.children())[:12] + [nn.AvgPool2d(1), Flatten()])   
                 )
                self.dropout=nn.Dropout(0.25)
                self.fc = nn.Linear(16384, 10)
            def forward(self, x):
                x1 = self.features(x)
                x2=self.dropout(x1)
                out=self.fc(x2)

                return out,x1

def predictive_entropy(image):
    predictions = []
    for fwd_pass in range(3):
        output,inter = net(image)
        #print(output)
        output = torch.nn.functional.softmax(output[0], dim=0)
        np_output = output.detach().cpu().numpy()
        if fwd_pass == 0:
            predictions = np_output
        else:
            predictions = np.vstack((predictions, np_output))
    #predictions=NormalizeData(predictions)
    epsilon = sys.float_info.min
    predictive_entropy = -np.sum( np.mean(predictions, axis=0) * np.log(np.mean(predictions, axis=0) + epsilon),
            axis=-1)
    
    #print(predictions)
    return predictive_entropy

PATH = './cifar_net.pth'
net = vgg()
net.load_state_dict(torch.load(PATH))
net.eval()

for module in net.modules():
    if module.__class__.__name__.startswith('Dropout'):
        module.train()

correct = 0
total = 0
subtotal =0
threshold=0.09
certain_correct = 0
certain_total = 0
uncertain_correct = 0
uncertain_total = 0
subtotal =0

threshold=0.21
with torch.no_grad():
    for image, label in testset:
        
        input_batch = image.unsqueeze(0) 
        uncertainty=predictive_entropy(input_batch)
        output,inter=net(input_batch)
        #print(output)
        _, predicted = torch.max(output.data,1)
        
        if uncertainty<threshold:
            certain_total+=1
            certain_correct+=(predicted==label)
            
            #print("certain total: %d, certain accuracy: %0.5f, uncertainty: %0.5f"%(certain_total, certain_correct/certain_total,uncertainty))
        else:
            uncertain_total+=1
            uncertain_correct+=(predicted==label)
            
            #print("uncertain total: %d, uncertain accuracy: %0.5f, uncertainty: %0.5f"%(uncertain_total, uncertain_correct/uncertain_total,uncertainty))
print("uncertain accuracy: %0.5f, certain accuracy: %0.5f"%(uncertain_correct/uncertain_total,certain_correct/certain_total))
print(uncertain_total, certain_total)
            

