import torch
from torchvision import transforms

### dataset

def replace_nans(data):
    """
    checks each channel of the image, the channels that are filled with NaN.
    if two channels are all NaN, replace with the third channel values.
    if one channel is all NaN, replace with the mean of the other channels
    """
    if len(data.shape) == 3:
        nan_channels = []
        val_channels = []

        for i in range(data.shape[0]):
            if torch.isnan(data[i,:,:]).all(): # all values are nan in the channel 
                nan_channels.append(i)
            else:
                val_channels.append(i)
            
        if len(nan_channels) == 1:
            data[nan_channels[0],:,:] = (data[val_channels[0],:,:] + data[val_channels[1],:,:]) / 2

        elif len(nan_channels) == 2:
            data[nan_channels[0],:,:] == data[val_channels[0],:,:]
            data[nan_channels[1],:,:] == data[val_channels[0],:,:]
    
    if torch.isnan(data).any():
        data[torch.isnan(data)] = 0.5
      
    return data


class ReplaceNans(object):
    def __call__(self, tensor):
        image = replace_nans(tensor)
        
        return tensor


def rescale(data, newmin, newmax): # https://stackoverflow.com/questions/33610825/normalization-in-image-processing
    shape = len(data.shape) 
    if shape == 3:
        for i in range(shape):
            data[i,:,:] = (data[i,:,:] - torch.min(data[i,:,:])) * (newmax - newmin)/(torch.max(data[i,:,:]) - torch.min(data[i,:,:]))  + newmin

    else:
        data = (data - torch.min(data)) * (newmax - newmin)/(torch.max(data) - torch.min(data))  + newmin

    return data


class Rescale(object):
    def __init__(self, newmin, newmax):
        self.newmin = newmin
        self.newmax = newmax

    def __call__(self, tensor):
        image = rescale(tensor, self.newmin, self.newmax)
        
        return tensor

### transforms

def transform_legacy(training):
    if training:
        return transforms.Compose([transforms.CenterCrop(150), Rescale(0.0,1.0), ReplaceNans(), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(10), transforms.Resize(224)])
    else: # evaluation
        return transforms.Compose([transforms.CenterCrop(150), Rescale(0.0,1.0), ReplaceNans(), transforms.Resize(224)])

def transform_deepscan(training):
    if training == False: # evaluation
        return transforms.Compose([Rescale(0.0,1.0), ReplaceNans(), transforms.Resize(224)])
