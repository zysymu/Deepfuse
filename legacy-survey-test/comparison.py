from astropy.io import fits
import numpy as np
import os
from torchvision import transforms
import matplotlib.pyplot as plt

def rescale(data, newmin, newmax):
    shape = len(data.shape)

    if shape == 3:
        for i in range(shape):
            print(i, np.min(data[i,:,:]))
            data[i,:,:] = (data[i,:,:] - np.min(data[i,:,:])) * (newmax - newmin)/(np.max(data[i,:,:]) - np.min(data[i,:,:]))  + newmin

    else:
        data = (data - np.min(data)) * (newmax - newmin)/(np.max(data) - np.min(data))  + newmin

    return data


def rescale_old(data, newmin, newmax): # https://stackoverflow.com/questions/33610825/normalization-in-image-processing
    shape = len(data.shape)

    if shape == 3:
        for i in range(shape):
            print(i, np.min(data[i,:,:]))
            data[i,:,:] = (data[i,:,:] - np.min(data[i,:,:])) * (newmax - newmin)/(np.max(data[i,:,:]) - np.min(data[i,:,:]))  + newmin

    else:
        data = (data - np.min(data)) * (newmax - newmin)/(np.max(data) - np.min(data))  + newmin

    return data

def standardize(data):
    data = (data - np.mean(data)) / np.std(data)
    return data

def visualize(image1, image2, image3, image4):
    fig, axs = plt.subplots(2, 2, figsize=(15,15))

    plt.axis('off')

    avg = np.mean(np.arcsinh(image1))
    axs[0, 0].imshow(np.arcsinh(image1), origin='lower', vmin=avg*0.995, vmax=avg*1.005, cmap="binary_r")    

    avg = np.mean(np.arcsinh(image2))
    axs[0, 1].imshow(np.arcsinh(image2), origin='lower', vmin=avg*0.995, vmax=avg*1.005, cmap="binary_r")

    avg = np.mean(np.arcsinh(image3))
    axs[1, 0].imshow(np.arcsinh(image3), origin='lower', vmin=avg*0.995, vmax=avg*1.005, cmap="binary_r")

    avg = np.mean(np.arcsinh(image4))
    axs[1, 1].imshow(np.arcsinh(image4), origin='lower', vmin=avg*0.995, vmax=avg*1.005, cmap="binary_r")


    plt.show()


#################################################3

legacy = '/home/marcostidball/Deepfuse/legacy-survey-test'
deepscan = '/home/marcostidball/Desktop/Deepfuse/candidates-iguess'

dset_min = []
dset_max = []

for i in range(24): 
    l = os.path.join(deepscan, f'{i+1}.fits')
    hdul_d = fits.open(l) # 3 channels
    img_d = hdul_d[1].data

    dset_min.append(np.min(img_d))
    dset_max.append(np.max(img_d))

dset = (min(dset_min), max(dset_max))

for i in range(24): 
    l = os.path.join(legacy, f'{i}.fits')
    d = os.path.join(deepscan, f'{i+1}.fits') 

    hdul_l = fits.open(l) # 3 channels
    img_l = hdul_l[0].data

    hdul_d = fits.open(d) # 1 channel
    img_d = hdul_d[1].data

    img_l = rescale(img_l, 0, 1) # 8,9 parece interessante 
    img_d = rescale(img_d, 0, 1) # 8,9 parece interessante 

    #img_l = standardize(img_l) # 8,9 parece interessante 
    #img_d = standardize(img_d) # 8,9 parece interessante 

    visualize(img_l[0,:,:], img_l[1,:,:], img_l[2,:,:], img_d)        
