import numpy as np
import cv2
from torchvision import transforms

from steps import *
from models import *
from dataloader import *

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])


def get_data(batch_size, parse_mode):
    
    flickr_loader = get_loader_flickr(transform=transform,
                                      batch_size=batch_size,
                                      mode='test',
                                      parse_mode=parse_mode)



def list2string(query_list):
    """
    Convert a list of strings to a space separated string
    """
    assert type(query_list) is list, "Lists accepted only"
    separator = ' '
    final_string = separator.join(query_list)
    return final_string


def rgb2gray(rgb):
    # Convert color numpy image to grayscale
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def tensor2img(tensor_image):
    # Convert each tensor image to numpy image
    img = tensor_image.permute(1, 2, 0)
    color_img = img.numpy()
    bw_img = rgb2gray(color_img)

    return color_img, bw_img


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


def master_processor(element):
    matchmap, caption = clip_list(element['matchmap'],
                                  element['caption'])
    color_image, bw_image = element['image']
    mask_list = list()
    for mmap in matchmap:
        mask = cv2.resize(mmap, dsize=(224,224))
        mask_list.append(mask)
    mask_list = np.stack(mask_list, axis = 0)
    return color_image, bw_image, mask_list, caption

