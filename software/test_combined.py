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

PATH = './cifar_net_new.pth'
net = vgg()
net.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
net.eval()

for module in net.modules():
    if module.__class__.__name__.startswith('Dropout'):
        module.train()

#intermediate output
class vgg_cloud(nn.Module):
            def __init__(self):
                super(vgg_cloud, self).__init__()
                self.features = nn.Sequential(
 		    *(list(original_model.features.children())[12:] + [nn.AdaptiveAvgPool2d(output_size=(7, 7)),Flatten()] + list(original_model.classifier.children())[:-1]) 
                 )
                self.fc = nn.Linear(4096, 10)
            def forward(self, x):
                x = self.features(x)
                x=self.fc(x)
                return x

class vgg_edge(nn.Module):
            def __init__(self):
                super(vgg_edge, self).__init__()
                self.features = nn.Sequential(
 		    *(list(original_model.features.children())[:12] + [nn.AvgPool2d(1), Flatten()])   
                 )
                self.dropout=nn.Dropout(0.25)
                self.fc = nn.Linear(16384, 10)
            def forward(self, x):
                x = self.features(x)
                x=self.dropout(x)
                x=self.fc(x)

                return x

PATH1 = './cifar_net_new.pth'
edge = vgg_edge()
edge.load_state_dict(torch.load(PATH1,map_location=torch.device('cpu')))
cloud= vgg_cloud()
class vgg_all(nn.Module):
            def __init__(self):
                super(vgg_all, self).__init__()
                self.features1 = nn.Sequential(
 		    *(list(edge.features.children())[:12]) 
                 )
                self.features2 = cloud
               
            def forward(self, x):
                x = self.features1(x)
                x = self.features2(x)
                return x

PATH2 = './cifar_net_19_freeze.pth'
big_net = vgg_all()

big_net.load_state_dict(torch.load(PATH2,map_location=torch.device('cpu')))

big_net.eval()


class vgg_real_cloud(nn.Module):
            def __init__(self):
                super(vgg_real_cloud, self).__init__()
                self.features = big_net.features2
            def forward(self, x):
                x = self.features(x)
                
                return x

final_cloud_model=vgg_real_cloud()
final_cloud_model.eval()

certain_correct = 0
certain_total = 1
uncertain_correct = 0
uncertain_total = 1
cloud_correct =0
print(net)
threshold=0.09
i=0
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
            
            ###
            send_x=inter.detach().numpy()
            
            data_input=pickle.dumps((send_x,(label,uncertain_total)), protocol=pickle.HIGHEST_PROTOCOL)
            inputs_ten=pickle.loads(data_input)
            inputs=torch.from_numpy(inputs_ten[0])
            output1=final_cloud_model(inputs)
            ###

            #output1=final_cloud_model(inter)

            _, predicted1 = torch.max(output1.data,1)
            cloud_correct+=(predicted1==label)
            
            print("certain_total: %d, uncertain total: %d, certain accuracy: %0.5f, uncertain accuracy: %0.5f, cloud accuracy: %0.5f"%(certain_total,uncertain_total,certain_correct/certain_total, uncertain_correct/uncertain_total,cloud_correct/uncertain_total))
            
