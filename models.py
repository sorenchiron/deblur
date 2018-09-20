import torch
import torchvision
import torchfun as tf
from torchvision import transforms
import numpy as np 
import random
from archs import DeblurNet,BasicBlock


cuda=False

seed=random.randint(0,100000)
random.seed(seed-1)
torch.manual_seed(seed)
torch.cuda.seed_all()
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark=False # set to true if network inputs are almost invariant

if cuda:
    torch.cuda.device(0)


dataset = torchvision.datasets.ImageFolder('dataset/ImageNet_test',
    transform=transforms.Compose([
        transforms.RandomRotation(10,expand=False),
        transforms.RandomResizedCrop(256,scale=(0.9,1),ratio=(0.999,1.01)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        tf.transforms.RandomGaussianBlur(0.02,0.01),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1, hue=0.01),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
        ])
    )

dataiter = torch.utils.data.DataLoader(dataset,
    batch_size=4,
    num_workers=4,
    pin_memory=cuda
    )

deblur = DeblurNet(BasicBlock,[2,2,2,2],
    filter_basis=32,
    activation=torch.nn.SELU)


