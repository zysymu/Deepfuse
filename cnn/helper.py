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
