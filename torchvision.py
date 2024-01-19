import torchvision
# from torchvision.datasets import ImageFolder
# from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
# import torchvision.transforms as transforms


transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        #transforms.Resize(128,128)
    ]
)

'''
Download the dataset from torchvision lib
'''
data = torchvision.datasets.CIFAR10(root='./data',train=True,transform=transform,download=True)

train_loader = DataLoader(dataset=data,batch_size=32,shuffle=True)

'''
When you have your custom data of image you can use this technique
'''

data = torchvision.datasets.ImageFolder(root='./data',transform=transform,)

train_loader = DataLoader(dataset=data,batch_size=32,shuffle=True)
