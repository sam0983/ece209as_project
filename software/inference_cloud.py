import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.optim as optim
import socket
import pickle
import json
from timeit import default_timer as timer


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

PATH = './cifar_net.pth'
edge = vgg_edge()
edge.load_state_dict(torch.load(PATH))
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

PATH = './cifar_net_19_freeze.pth'
big_net = vgg_all()
big_net.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
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
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
print(final_cloud_model)
i=0
TCP_IP1 =''
TCP_PORT1 = 5006
BUFFER_SIZE = 4096
 
s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP1, TCP_PORT1))
s.listen()
with torch.no_grad():
    

    while 1:
        #print('Waiting for Raspberry 1')
 
        
        
        

        conn, addr = s.accept()
        data=[]
        #print ('Raspberry Device:',addr)
        while 1:
            tensor = conn.recv(BUFFER_SIZE)
            if not tensor: break
            data.append(tensor)
            
            
        inputs_ten=pickle.loads(b"".join(data))
        
        
        conn.close()
        
        inputs=torch.from_numpy(inputs_ten[0])
        
        output = final_cloud_model(inputs)
        #print(output)
        
        _, predicted = torch.max(output.data,1)
        total+=1
        correct+=(inputs_ten[1][0]==predicted)
        #print(total)
        print("total: %d, supposed total: %d, accuracy: %0.5f, TF: %d"%(total, inputs_ten[1][1],correct/total,inputs_ten[1][0]==predicted))
        
        
