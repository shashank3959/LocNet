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


def load_data(batch_size, parse_mode='phrase'):
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


def list2string(query_list):
    """
    Convert a list of strings to a space separated string
    """
    assert type(query_list) is list, "Lists accepted only"
    separator = ' '
    final_string = separator.join(query_list)
    return final_string



def clip_list(query_list, ann_id, data, parse_mode):
    if parse_mode == 'phrase':
        actual_length = len(data[ann_id]['parsed_caption'])
        caption = data[ann_id]['parsed_caption']
    else:
        actual_length = len(data[ann_id]['tok_sent'])
        caption = data[ann_id]['tok_sent']
    start = actual_length
    del query_list[(start + 1):]
    del query_list[0]
    return query_list, caption


def master_processor(element, ):
    co_loc_map, ann_id = element['co_loc_map'], elemenet['caption']

    color_image, bw_image = element['image']
    mask_list = list()
    for mmap in matchmap:
        mask = cv2.resize(mmap, dsize=(224,224))
        mask_list.append(mask)
    mask_list = np.stack(mask_list, axis = 0)
    return color_image, bw_image, mask_list, caption

