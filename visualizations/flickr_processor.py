import numpy as np
import cv2
from torchvision import transforms
import json

from steps.utils import *
from dataloader import get_loader_flickr
from visualize_utils import *


path_to_file = '../data/flickr_30kentities/annotations_flickr/Sentences/test/data.json'
f = open(path_to_file, encoding='utf-8', mode='r')
data = json.load(f)
print('Loaded Flickr annotation data!')
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])


def load_data(batch_size, parse_mode):
    """
    Loads data from test fold of flickr dataset using flickr_loader
    :param parse_mode: decides whether the captions should be parsed 
    :return: image tensor, caption glove tensor and tuple of caption ids
    """
    flickr_loader = get_loader_flickr(transform=transform, batch_size=batch_size,
                                      mode='test', parse_mode=parse_mode)

    for batch in flickr_loader:
        (image, caption_glove, caption, ids) = batch

    return image, caption_glove, ids


def gen_data_element(index, batch_size=1, parse_mode='phrase',
                     model_path='..saved_models/checkpoint.pth.tar'):
    """
    Generates a dictionary of data elements from flickr dataset
    :param index: choose data from a batch using this index
    :param batch_size: choose to select a specific batch size
    :param parse_mode: choose whether to parse flickr captions based on entities
    :param model_path: path to trained model, if exists.
    :return element: element is a dictionary containing the following elements
    :return element['image']['col']: 3 channel color image
    :return element['image']['bw']: single channel bw image
    :return element['caption']: tokenized and clipped caption depending on parse_mode
    :return element['coloc_map']: resized and clipped coloc_map masks for the given data point
    """
    assert index < batch_size, "Index out of range"

    image_model, caption_model = get_models(model_path)
    image_tensor, caption_glove, ids = load_data(batch, parse_mode=parse_mode)
    coloc_maps = gen_coloc_maps(image_model, caption_model,
                                image_tensor, caption_glove)
    raw_element = fetch_data(index, coloc_maps, image_tensor, ids)

    element = flickr_element_processor(raw_element, parse_mode, data)

    return element


