import torchvision.models as models
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.optim as optim
import copy
import time
import numpy as np
import pandas as pd
from helper_classifier import freezer, make_plots, make_cm

class Classifier(object):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = xm.xla_device()

    def __init__(self, model_name, finetune):
        """
        pass
        """
        print('Using device:', Classifier.device)

        self.model_name = model_name
        self.finetune = finetune

        if model_name == 'resnet50':
            model = models.resnet50(pretrained=True) # loads model                
            blocks = ['layer1', 'layer2', 'layer3', 'layer4']
                
            model = freezer(model, finetune, blocks)

            num_ftrs = model.fc.in_features # gets last layer's features
            model.fc = nn.Sequential(nn.Linear(num_ftrs, 2)) # changes output to 2     

        elif model_name == 'vgg16':
            model = models.vgg16_bn(pretrained=True) 
            blocks = None
                
            model = freezer(model, finetune, blocks)

            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, 2))

        elif model_name == 'vgg19':
            model = models.vgg19_bn(pretrained=True) 
            blocks = None
                
            model = freezer(model, finetune, blocks)

            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, 2))

        elif model_name == 'densenet':
            model = models.densenet169(pretrained=True) 
            blocks = None
                
            model = freezer(model, finetune, blocks)

            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(nn.Linear(num_ftrs, 2))

        elif model_name.split('-')[0] == 'efficientnet':
            n = model_name.split('-')[1]
            model = EfficientNet.from_pretrained(f'efficientnet-{n}')
            blocks = None
                
            model = freezer(model, finetune, blocks)

            num_ftrs = model._fc.in_features
            model._fc = nn.Sequential(nn.Linear(num_ftrs, 2))

        else:
            print("Make sure the `model` parameter is equal to one of the following: 'resnet152', 'resnet50', 'vgg16', 'vgg19', 'densenet' or 'efficientnet-bN', where N = 1,2,...,7")
            print("Make sure the `finetune` parameter works for the model that is being used")

        self.model = model.to(Classifier.device) # send model to gpu
    
    def fit(self, dataloaders, epochs, checkpoint_path):
        """
        pass
        """
        criterion = nn.CrossEntropyLoss()

        if not os.path.isfile(checkpoint_path): # if checkpoint doesn't exist
            optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            start_epoch = 0

            best_model_wts = copy.deepcopy(self.model.state_dict())

            history = {'train_loss': [],
                    'train_acc': [],
                    'val_loss': [],
                    'val_acc': [],
                    'best_loss': np.Inf,
                    'best_acc': 0.0,
                    'best_epoch': 0}   

        else:
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            optimizer = optim.Adam(self.model.parameters())
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1

            best_model_wts = checkpoint['best_model_wts_state_dict']
            
            history = checkpoint['history']

        since = time.time()

        for epoch in range(start_epoch, epochs):

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data
                loop = tqdm(dataloaders[phase], total=len(dataloaders[phase]), position=0, leave=True)
                for inputs, labels in (loop):
                    inputs = inputs.to(Classifier.device)
                    labels = labels.to(Classifier.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            #xm.optimizer_step(optimizer, barrier=True)  # Note: Cloud TPU-specific code!

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # update tqdm
                    loop.set_description(f'Epoch {epoch}/{epochs} [{phase}]')
                
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
                if phase == 'val' and epoch_loss <= history['best_loss']:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(history['best_loss'],epoch_loss))
                    history['best_loss'] = epoch_loss
                    history['best_acc'] = epoch_acc 
                    history['best_epoch'] = epoch
                    best_model_wts = copy.deepcopy(self.model.state_dict())

                # save checkpoint
                torch.save({'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch,
                            'history': history,
                            'best_model_wts_state_dict': best_model_wts}, checkpoint_path)

            print()

        time_elapsed = time.time() - since
        print('\n Training complete in {:.0f}m {:.0f}s \n'.format(time_elapsed // 60, time_elapsed % 60))
        print('\n Best epoch: {}; Loss: {:4f}, Acc: {:4f} \n'.format(history['best_epoch'], history['best_loss'], history['best_acc']))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        self.history = history
        
        self.checkpoint_path = checkpoint_path

    def metrics((self, dir_path, checkpoint_path, testloader):
        """
        pass
        """
        last_checkpoint = torch.load(checkpoint_path)
        self.history = last_checkpoint['history']

        # plots
        make_plots(self.history, dir_path)

        true_labels, predictions = self.predict(testloader, output_path=None)

        make_cm(true_labels, predictions, dir_path)
    
        #csv
        best_epoch = self.history['best_epoch']
        epochs = len(self.history['train_loss'])

        df = pd.DataFrame({'model_name':[self.model_name],
                            'finetune': [self.finetune],
                            'true_negatives':[tn],
                            'false_positives':[fp],
                            'false_negatives':[fn],
                            'true_positives':[tp],
                            'best_epoch':[best_epoch],
                            'epochs': [epochs]})
        
        df.to_csv(os.path.join(dir_path, 'stats.csv'))


    def metrics_candidates(self, dir_path, checkpoint_path, testloader, candidatesloader_legacy, candidatesloader_deepscan):
        """
        pass
        """
        last_checkpoint = torch.load(checkpoint_path)
        self.history = last_checkpoint['history']

        make_plots(self.history, dir_path)

        true_labels, predictions = self.predict(testloader, output_path=None)

        tn, fp, fn, tp = make_cm(true_labels, predictions, dir_path)

        #csv
        best_epoch = self.history['best_epoch']
        epochs = len(self.history['train_loss'])

        # candidates
        true_labels_legacy, predictions_legacy = self.predict(candidatesloader_legacy, output_path=None)
        true_labels_deepscan, predictions_deepscan = self.predict(candidatesloader_deepscan, output_path=None)

        assert true_labels_legacy == true_labels_deepscan
        gals_legacy = np.where(predictions_legacy>0.5, 1, 0) 
        gals_deepscan = np.where(true_labels_deepscan>0.5, 1, 0) 
            
        gals_legacy = gals_legacy.sum()
        gals_deepscan = gals_deepscan.sum()

        non = len(gals_legacy) - gals_legacy

        df = pd.DataFrame({'model_name':[self.model_name],
                           'finetune': [self.finetune],
                           'true_negatives':[tn],
                           'false_positives':[fp],
                           'false_negatives':[fn],
                           'true_positives':[tp],
                           'galaxies_legacy': [gals_legacy],
                           'galaxies_deepscan': [gals_deepscan],
                           'non_galaxies':[non],
                           'best_epoch':[best_epoch],
                           'epochs': [epochs]})
        
        df.to_csv(os.path.join(dir_path, 'stats.csv'))

    def save(self, filepath, save_story=True): # use .pt extension
        torch.save(self.model.state_dict(), filepath)
        print('Model saved to {} !'.format(filepath))

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        print('Model loaded from {} !'.format(filepath))

    def predict(self, dataloader, output_path):
        self.model.eval() # set dropout and batch normalization layers to evaluation mode
        
        with torch.no_grad():
            if output_path == None:
                
                true_labels = []
                predictions = []

                for inputs, labels in dataloader:
                    inputs = inputs.to(Classifier.device)
                    labels = labels.to(Classifier.device)

                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    preds = preds.cpu().numpy()
                    labels = labels.cpu().numpy()

                    true_labels.append(labels)
                    predictions.append(preds)

                true_labels = np.concatenate(true_labels)
                predictions = np.concatenate(predictions)

                return true_labels, predictions

            else:

                total_filenames = []
                predictions = []

                for inputs, filenames in dataloader:
                    inputs = inputs.to(Classifier.device)

                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    preds = preds.cpu().numpy()

                    total_filenames.append(filenames)
                    predictions.append(preds)

                total_filenames = np.array(total_filenames)
                total_filenames = np.concatenate(total_filenames)
                predictions = np.concatenate(predictions)
                
                print(total_filenames, predictions)

                df = pd.DataFrame({'file': total_filenames, 'label': predictions})
                df.to_csv(output_path, index=False)

    def __repr__(self):
        return str(self.model_name, self.finetune)
