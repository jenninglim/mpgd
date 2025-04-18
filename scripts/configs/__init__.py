import configs.cifar
import configs.mnist
import torchvision
from torchvision import transforms

DATASETS = {'mnist' : torchvision.datasets.MNIST,
            'cifar' : torchvision.datasets.CIFAR10,}