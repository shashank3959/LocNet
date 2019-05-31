"""
Will have general utils functions that will be used  by both
1. flickr_processor.py
2. coco_processor.py

First built keeping flickr_processor in mind.

Functions included:
	1. load model
	2. generate matchmap list
	3. fetch specific image, matchmap and caption/ID with index
	4. clip matchmap and caption
	5. process images
"""

import torch
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from steps.utils import matchmap_generate
from models import VGG19, LSTMBranch


def get_models(model_path='..saved_models/checkpoint.pth.tar'):
    '''
    Load pre-trained model
    :return: pretrained image and caption model
    '''
    image_model = VGG19(pretrained=True)
    caption_model = LSTMBranch()

    if not os.path.exists(model_path):
        print("Not using trained models")
        return image_model, caption_model

    checkpoint = torch.load(model_path, map_location='cpu')
    image_model.load_state_dict(checkpoint['image_model'])
    caption_model.load_state_dict(checkpoint['caption_model'])
    print('Loaded pretrained models')
    return image_model, caption_model


def gen_coloc_maps(image_model, caption_model,
                   image_tensor, caption_glove):
    """
    Generate a list of numpy match-maps from a batch of data
    :param models and tensors:Caption glove and image tensors required 
    generate a list of co_localization maps.
    :return: A list of co-localization maps. Each map is a numpy ndarray
    """
    image_op = image_model(image_tensor)
    caption_op = caption_model(caption_glove)
    batch_size = image_model.size(0)
    coloc_maps = list()

    for i in tqdm(np.arange(batch_size)):
        coloc = matchmap_generate(image_op[i], caption_op[i])
        mm = coloc.detach().numpy()
        coloc_maps.append(mm)

    return coloc_maps


def fetch_data(index, coloc_maps, image_tensor, captions):
    """
    Parse all necessary data based on index into a dictionary
    :return element: Dictionary containing three items:
    1. image tensor at index-th index
    2. caption/Caption ID depending on dataset being used
    3. coloc_map: untrimmed co_loc map at index position.
    """
    element = dict()
    element['coloc_map'] = coloc_maps[index]
    element['image'] = image_tensor[index]
    element['caption'] = captions[index]  # In case of flickr, AnnID

    return element


def rgb2gray(rgb):
    """
    Convert color numpy image to grayscale
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def tensor2img(tensor_image):
    """
    Convert each tensor image to numpy image
    :return image: Dictionary of two elements
    :return image['col']: Numpy 3 channel color image
    :return image['bw']: Numpy single channel BW image
    """
    img = tensor_image.permute(1, 2, 0)
    color_img = img.numpy()
    bw_img = rgb2gray(color_img)
    image = {"color": color_img, "bw": bw_img}

    return image


def coloc_map_processor(coloc_map):
    """
    Process a list of coloc maps
    :param coloc_map: List of numpy matrices. Clipped coloc map.
    :return mask_list: numpy stack of resized masks to be visualized
    """
    mask_list = list()
    for frame in coloc_map:
        mask = cv2.resize(frame, dsize=(224, 224))
        mask_list.append(mask)
    mask_list = np.stack(mask_list, axis=0)
    return mask_list


def flickr_element_processor(element, parse_mode, data):
    """
    Clips co_loc maps and captions for FLICKR data points
    :param parse_mode: Decides which caption to fetch using ann_id
    :param data: dictionary containing data of all captions
    :return: return dict element with clipped & resized co_loc masks
    clipped co_loc maps and captions have no '<start>' and '<end>'
    Processed image has been converted to np images ready to be plotted
    """
    caption_data = data[element['caption']]
    coloc_map = element['coloc_map']
    if parse_mode == 'phrase':
        caption = caption_data['parsed_caption']
    else:
        caption = caption_data['tok_sent']
    actual_length = len(caption)
    del coloc_map[(actual_length+1):]
    del coloc_map[0]

    coloc_map = coloc_map_processor(coloc_map)

    element['coloc_map'] = coloc_map
    element['caption'] = caption
    element['image'] = tensor2img(element['image'])

    return element


def coco_element_processor(element):
    """
    Clips co_loc maps and captions for COCO data points
    :return: return dict element with clipped & resized co_loc masks
    clipped co_loc maps and captions have no '<start>' and '<end>'
    Processed image has been converted to np images ready to be plotted
    """
    caption = element['caption']
    coloc_map = element['coloc_map']
    start = caption.index('<end>')
    del [caption[start:]]
    del [coloc_map[start:]]
    del [caption[0]]
    del [coloc_map[0]]

    coloc_map = coloc_map_processor(coloc_map)

    element['coloc_map'] = coloc_map
    element['caption'] = caption
    element['image'] = tensor2img(element['image'])

    return element


