import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

classes = ('circle', 'square', 'triangle')

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# trainset = torchvision.datasets.ImageFolder("./train_shapes", transform=transform)
trainset = torchvision.datasets.ImageFolder("./train_shapes")
trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True, num_workers=2)

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

print(len(images[0][0]),"x",len(images[0][0]), "x",len(images[0]))

# # show images
imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(10)))
# print(labels)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=1)
        self.pool = nn.MaxPool2d(1, 1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 24 * 24 , 30)
        self.fc2 = nn.Linear(30, 10)
        self.fc3 = nn.Linear(10, 3)

    def forward(self, x):
        #print("size: ",x.size())
        x = self.pool(F.relu(self.conv1(x)))
        #print("size: ",x.size())
        x = self.pool(F.relu(self.conv2(x)))
        #print("size: ",x.size())
        x = x.view(-1, 16 * 24 * 24)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.4)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        #print(i)

        #print(inputs)
        #zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #print(outputs)
        #print(labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #print("Loss", loss.item())

        # print statistics
        running_loss += loss.item()

        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1))
        running_loss = 0.0

print('Finished Training')

PATH = './train_shapes_net.pth'
torch.save(net.state_dict(), PATH)
print("Done")


testset = torchvision.datasets.ImageFolder(root='./test_shapes', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True, num_workers=2)

dataiter = iter(testloader)
images, labels = dataiter.next()
# # print images
# imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(10)))

net = Net()
net.load_state_dict(torch.load("./train_shapes_net.pth"))

outputs = net(images)

print("")
print(outputs)
print("")

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(10)))


print("")
print(classes)
