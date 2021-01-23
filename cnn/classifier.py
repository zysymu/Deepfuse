import torchvision.models as models
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
import matplotlib.pyplot as plt
from helper import freezer, blocks_mapper


class Classifier(object):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, model_name, finetune): # INPUT SIZE: 224 X 224
    """
    pass
    """
        print('Using device:', self.device)

        self.model_name = model_name
        self.finetune = finetune

        if model_name == 'resnet152':
            model = models.resnet152(pretrained=True) # loads model                
            blocks = ['layer1', 'layer2', 'layer3', 'layer4']
                
            model = freezer(model, finetune, blocks)
                
            num_ftrs = model.fc.in_features # gets last layer's features
            model.fc = nn.Linear(num_ftrs, 2) # changes output to 2     

            self.model = model.to(self.device) # sends model to gpu

        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            blocks = ['layer1', 'layer2', 'layer3', 'layer4']
                
            model = freezer(model, finetune, blocks)

            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)

            self.model = model.to(self.device)

        elif model_name == 'vgg16':
            model = models.vgg16_bn(pretrained=True) 
            blocks = None
                
            model = freezer(model, finetune, blocks)

            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, 2))

            self.model = model.to(self.device) 

        elif model_name == 'vgg19':
            model = models.vgg19_bn(pretrained=True) 
            blocks = None
                
            model = freezer(model, finetune, blocks)

            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, 2))

            self.model = model.to(self.device) 

        elif model_name == 'densenet':
            model = models.densenet169(pretrained=True) 
            blocks = None
                
            model = freezer(model, finetune, blocks)

            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(nn.Linear(num_ftrs, 2))

            self.model = model.to(self.device) 

        elif model_name.split('-')[0] == 'efficientnet':
            n = model_name.split('-')[1]
            model = EfficientNet.from_pretrained(f'efficientnet-{n}')
            blocks = None
                
            model = freezer(model, finetune, blocks)

            num_ftrs = model._fc.in_features
            model._fc = nn.Sequential(nn.Linear(num_ftrs, 2))

            self.model = model.to(self.device) 
        
        else:
            print("Make sure the `model` parameter is equal to one of the following: 'resnet152', 'resnet50', 'vgg16', 'vgg19', 'densenet' or 'efficientnet-bN', where N = 1,2,...,7")
            print("Make sure the `finetune` parameter works for the model that is being used")

    def fit(self, dataloaders, epochs, criterion, optimizer, scheduler):
        """
        pass
        """    
        history = {'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'best_loss': 0.0,
                'best_acc': 0.0,
                'best_epoch': 0}

        best_model_wts = copy.deepcopy(self.model.state_dict())    
        best_loss = np.Inf

        since = time.time()

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data
                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                if phase == 'train':
                    history['train_loss'].append(epoch_loss)
                    history['train_acc'].append(epoch_acc)
                    
                elif phase == 'val':
                    history['val_loss'].append(epoch_loss)
                    history['val_acc'].append(epoch_acc)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_loss <= best_loss:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(best_loss,epoch_loss))
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best epoch: {}; Loss: {:4f}, Acc: {:4f}'.format(best_epoch, best_loss, best_acc))

        # load best model weights
        self.model = self.model.load_state_dict(best_model_wts)

        history['best_loss'] = best_loss
        history['best_acc'] = best_acc
        history['best_epoch'] = best_epoch
        self.history = history

        return self.model, self.history  

    def metrics(self, dir_path):
        """
        pass
        """
        os.mkdir(dir_path)

        """
        usar a melhor epoca e melhor loss e marcar esses pontos nos graficos, 
        talvez usando aquelas mesmas flechas que usei no trabalho de astro.

        se pa fazer grafico de loss e accuracy juntos mesmo, tipo no tanoglidis
        e limitar os valores pra ficarem com 1 no maximo no y axis.
        """

    def save(self, path): # use .pt extension
        torch.save(self.model.state_dict(), path)
        print('Model saved to {} !'.format(path))

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def predict(self, dataloader, output_path):
        self.model.eval() # set dropout and batch normalization layers to evaluation mode

    def __repr__(self):
        return str(self.model_name, self.finetune)
