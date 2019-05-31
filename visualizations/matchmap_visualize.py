from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import json

from steps import *
from models import *
from dataloader import *

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

def get_data(batch_size, dataset, parse_mode):
    '''
    Load dataset in required fashion
    :param batch_size: batch size of data to be loaded
    :param dataset: which dataset to use: flickr or mscoco
    :param parse_mode: parse_mode for flickr
    :return: return tensors image and caption glove along with caption
    '''
    if dataset == 'flickr':
        flickr_loader = get_loader_flickr(transform=transform,
                                          batch_size=batch_size,
                                          mode='test',
                                          parse_mode=parse_mode)
        for batch in flickr_loader:
            image, caption_glove, ann_id = batch[0], batch[1], batch[2]
        return image, caption_glove, ann_id

    else:
        assert parse_mode != 'phrase', "MSCOCO doesn't support phrase parsing"
        coco_loader = get_loader_coco(transform=transform,
                                      batch_size=batch_size,
                                      mode='val',
                                      fetch_mode='retrieval')
        for batch in coco_loader:
            image, caption_glove, caption = batch[0], batch[1], batch[2]

        return image, caption_glove, caption


def get_models(model_path='../model_best.pth.tar'):
    '''
    Load pre-trained model
    :return: pretrained image and caption model
    '''
    image_model = VGG19(pretrained=True)
    caption_model = LSTMBranch()

    if not os.path.exists(model_path):
        print("Not using trained models")
        return image_model, caption_model
    print("Loading pretrained model")
    checkpoint = torch.load(model_path, map_location='cpu')
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    image_model.load_state_dict(checkpoint['image_model'])
    caption_model.load_state_dict(checkpoint['caption_model'])
    print('Loaded models')
    return image_model, caption_model


def gen_matchmap_list(image_model, caption_model,
                      image_tensor, caption_glove,
                      batch_size):
    """
    Generate a list of numpy match-maps from a batch of data
    """
    image_op = image_model(image_tensor)
    caption_op = caption_model(caption_glove)
    matchmap_list = list()

    for i in tqdm(np.arange(batch_size)):
        matchmap = matchmap_generate(image_op[i], caption_op[i])
        mm = matchmap.detach().numpy()
        matchmap_list.append(mm)

    return matchmap_list

def pre_process_matchmaps(matchmap_list):
    pass

def see_results(dataset='flickr', batch_size=1, parse_mode='phrase'):
    path_to_file = '../data/flickr_30kentities/annotations_flickr/Sentences/test/data.json'
    if dataset == 'flickr':
        image, caption_glove, ann_id = get_data(batch_size,
                                              dataset,
                                              parse_mode)
        print('Loaded data!')
        f = open(path_to_file, encoding='utf-8', mode='r')
        data = json.load(f)
        print('Loaded annotation data!')


    image_model, caption_model = get_models(model_path='')
    matchmap_list = gen_matchmap_list(image_model, caption_model,
                                      image, caption_glove,
                                      batch_size)



    pass
