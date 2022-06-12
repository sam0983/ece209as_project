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
               
                self.fc = nn.Linear(16384, 10)
            def forward(self, x):
                x1 = self.features(x)
                
                out=self.fc(x1)

                return out,x1

def predictive_entropy(image,net1,net2,net3):
    predictions = []
    ensembles=[net1,net2,net3]
    for i,model in enumerate(ensembles):
        output,inter = model(image)
        #print(output)
        output = torch.nn.functional.softmax(output[0], dim=0)
        np_output = output.detach().cpu().numpy()
        if i == 0:
            predictions = np_output
        else:
            predictions = np.vstack((predictions, np_output))
    #predictions=NormalizeData(predictions)
    epsilon = sys.float_info.min
    predictive_entropy = -np.sum( np.mean(predictions, axis=0) * np.log(np.mean(predictions, axis=0) + epsilon),
            axis=-1)
    
    #print(predictions)
    return predictive_entropy

PATH1 = './cifar_net_ens_1.pth'
PATH2 = './cifar_net_ens_2.pth'
PATH3 = './cifar_net_ens_3.pth'
net1 = vgg()
net2 =vgg()
net3 = vgg()
net1.load_state_dict(torch.load(PATH1,map_location=torch.device('cpu')))
net1.eval()
net2.load_state_dict(torch.load(PATH2,map_location=torch.device('cpu')))
net2.eval()
net3.load_state_dict(torch.load(PATH3,map_location=torch.device('cpu')))
net3.eval()

correct = 0
total = 0
subtotal =0
threshold=0.09
certain_correct = 0
certain_total = 1
uncertain_correct = 0
uncertain_total = 1
subtotal =0

threshold=0.28
with torch.no_grad():
    for image, label in testset:
        
        input_batch = image.unsqueeze(0) 
        uncertainty=predictive_entropy(input_batch,net1,net2,net3)
        output,inter=net1(input_batch)
        #print(output)
        _, predicted = torch.max(output.data,1)
        
        if uncertainty<threshold:
            certain_total+=1
            certain_correct+=(predicted==label)
            
            #print("certain total: %d, uncertain total: %d, certain accuracy: %0.5f, uncertain accuracy: %0.5f"%(certain_total,uncertain_total, certain_correct/certain_total,uncertain_correct/uncertain_total))
        else:
            uncertain_total+=1
            uncertain_correct+=(predicted==label)
            
            #print("uncertain total: %d, uncertain accuracy: %0.5f, uncertainty: %0.5f"%(uncertain_total, uncertain_correct/uncertain_total,uncertainty))
print("uncertain accuracy: %0.5f, certain accuracy: %0.5f"%(uncertain_correct/uncertain_total,certain_correct/certain_total))
print(uncertain_total, certain_total)
            

