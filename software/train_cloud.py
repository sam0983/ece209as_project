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

big_net=vgg_all()
counter=0
print(big_net)
"""
for name, param in big_net.named_parameters(): 
    if name.startswith("features1"):
        param.requires_grad=False
        print(name)
"""
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(big_net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = big_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print('Finished Training')
PATH = './cifar_net.pth'
torch.save(big_net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = dataiter.next()

big_net = vgg_all()
big_net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = big_net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
