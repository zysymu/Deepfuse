import torchvision.models as models
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import helper.freezer as freezer

class Classifier(object):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, model, finetune):
        print(self.device)
        if model == 'resnet152':
            model = models.resnet152(pretrained=True) # loads model                
            blocks = ['layer1', 'layer2', 'layer3', 'layer4']
                
            model = freezer(model, finetune, blocks)
                
            num_ftrs = model.fc.in_features # gets last layer's features
            model.fc = nn.Linear(num_ftrs, 2) # changes output to 2     

            self.model = model.to(self.device) # sends model to gpu

        elif model == 'resnet50':
            model = models.resnet50(pretrained=True)
            blocks = ['layer1', 'layer2', 'layer3', 'layer4']
                
            model = freezer(model, finetune, blocks)

            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)

            self.model = model.to(self.device)

        elif model == 'vgg16':
            model = models.vgg16_bn(pretrained=True) 
            blocks = None
                
            model = freezer(model, finetune, blocks)

            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, 2))

            self.model = model.to(self.device) 

        elif model == 'vgg19':
            model = models.vgg19_bn(pretrained=True) 
            blocks = None
                
            model = freezer(model, finetune, blocks)

            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, 2))

            self.model = model.to(self.device) 

        elif model == 'densenet':
            model = models.densenet169(pretrained=True) 
            blocks = None
                
            model = freezer(model, finetune, blocks)

            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(nn.Linear(num_ftrs, 2))

            self.model = model.to(self.device) 

        elif model.split('-')[0] == 'efficientnet':
            n = model.split('-')[1]
            model = EfficientNet.from_pretrained(f'efficientnet-{n}')
            blocks = None
                
            model = freezer(model, finetune, blocks)

            num_ftrs = model._fc.in_features
            model._fc = nn.Sequential(nn.Linear(num_ftrs, 2))

            self.model = model.to(self.device) 
        
        else:
            print("Make sure the `model` parameter is equal to one of the following: 'resnet152', 'resnet50', 'vgg16', 'vgg19', 'densenet' or 'efficientnet-bN', where N = 1,2,...,7")
            print("Make sure the `finetune` parameter works for the model that is being used")

    def fit(self, dataloader, epochs, batch_size, optimizer_lr, patience):
        pass

    def eval(self, path): # maybe being together with the fit method
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def predict(self, dataloader, output_path):
        pass

    def __repr__(self):
        return str(self.model)