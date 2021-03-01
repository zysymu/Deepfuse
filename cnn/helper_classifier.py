import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

def freezer(model, finetune, blocks):
    """
    Freezes a certain part of the model.
    """
    try:
        assert (finetune >= 0) or (finetune == True) or (finetune == False)
    except AssertionError:
        print("Make sure the `finetune` parameter is >= 0, True or False")
    

    # retrain the whole network:
    if finetune == True:
        pass

    # feature extractor, freeze layers and only trains the fully connected layer:
    elif finetune == False: 
        for param in model.parameters(): # freeze layers
            param.requires_grad = False

    # freeze only certain layers:
    elif int(finetune) >= 0: 
        if blocks != None:
            blocks_map = blocks_mapper(blocks)
            
            try:
                block = blocks_map[int(finetune)]
            except:
                print("Make sure the `finetune` parameter is valid for this model")

            for name, child in model.named_children():
                if name not in [block]:
                    for param in child.parameters(): # freeze
                            param.requires_grad = False

                # freezes up to the `finetune` layer:
                # if `finetune = 'layer4'`, we'll freeze start -> layer1 -> layer2 ->
                # layer3 and let 'layer4 and everything boeyond it trainable
                elif name in [block]: 
                    break

    return model


def blocks_mapper(blocks):
    blocks_map = {i:name for i, name in enumerate(blocks)}
    
    return blocks_map


def make_plots(history, dir_path):
    plt.style.use('ggplot')

    os.makedirs(dir_path, exist_ok=True)

    # loss and accuracy
    epochs_train = range(len(history['train_loss']))
    epochs_val = range(len(history['val_loss']))

    plt.figure(figsize=(7,7), dpi=200)
    plt.plot(epochs_train, history['train_loss'], c='mediumblue', label='Training loss', linestyle='--')
    plt.plot(epochs_train, history['train_acc'], c='darkred', label='Training accuracy', linestyle='--')
    
    plt.plot(epochs_val, history['val_loss'], c='dodgerblue', label='Validation loss')
    plt.plot(epochs_val, history['val_acc'], c='darkorange', label='Validation accuracy')

    best_epoch = history['best_epoch']
    plt.scatter(history['best_epoch'], history['best_loss'], c='black', marker='*', label=f'Best loss @ epoch {best_epoch}', zorder=10)

    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')

    plt.xlim(0, epochs_train[-1])
    plt.ylim(0,1)
    plt.legend()
    plt.title('Loss/Accuracy over Epochs')

    plt.savefig(os.path.join(dir_path, 'loss_acc.pdf'))    

def make_cm(true_labels, predictions, dir_path, p=0.5):
    p = 0.5
    cm = confusion_matrix(true_labels, predictions > p)
    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    group_percentages = ['({0:.2%})'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    tn, fp, fn, tp = cm.ravel()
        
    plt.figure(figsize=(7,7), dpi=200)
    sns.heatmap(cm, annot=labels, fmt="")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(dir_path, 'confusion_matrix.pdf'))

    return tn, fp, fn, tp
