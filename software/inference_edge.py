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
import time

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
                
                x1 = self.features[:12](x)
                x2 = self.features[12:](x1)
                x3=self.dropout(x2)
                out=self.fc(x3)

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



certain_correct = 0
certain_total = 0
uncertain_correct = 0
uncertain_total = 0
print(net)
threshold=0.09
i=0
TCP_IP2 = '127.0.0.1'
TCP_PORT2 = 5006

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
            
            print("uncertain total: %d, uncertain accuracy: %0.5f, uncertainty: %0.5f"%(uncertain_total, uncertain_correct/uncertain_total,uncertainty))
            s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((TCP_IP2,TCP_PORT2))
            #print(inter)
            send_x=inter.detach().numpy()
            h=input_batch.detach().numpy()
            print(send_x.size * send_x.itemsize,h.size * h.itemsize)
            
            data_input=pickle.dumps((send_x,(label,uncertain_total)), protocol=pickle.HIGHEST_PROTOCOL)
            s.sendall(data_input)
            s.close
            
            """
            ####label
            s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            TCP_PORT2 = 5005-i
            s.connect((TCP_IP2,TCP_PORT2))
            send_label=label.detach().numpy()
            data_input=pickle.dumps(send_label, protocol=pickle.HIGHEST_PROTOCOL)
            s.sendall(data_input)
            s.close
            ####
            """
            print("data sent to server")
            #time.sleep(5.0)

