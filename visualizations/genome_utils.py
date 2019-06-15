import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

from steps.utils import matchmap_generate
from models import VGG19, LSTMBranch

import sys

path_to_dataloader = '../'
sys.path.append(path_to_dataloader)

from dataloader import *


def genome_load_data(batch_size, transform, mode='train'):
    genome_loader = get_loader_genome(transform=transform,
                                      mode='train',
                                      batch_size=batch_size)

    for batch in genome_loader:
        (image_tensor, caption_glove, ann_id) = batch

    return genome_loader, image_tensor, caption_glove, ann_id


def get_models_genome(model_path):
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


def gen_coloc_maps_matchmap(image_model, caption_model,
                   image_tensor, caption_glove):
    """
    This gen_coloc processes a phrase and then averages matchmaps for each word
    Generate a list of numpy match-maps from a batch of data
    :param models and tensors:Caption glove and image tensors required 
    generate a list of co_localization maps.
    :return: A list of co-localization maps. Each map is a numpy ndarray
    """
    image_op = image_model(image_tensor)
    caption_op = caption_model(caption_glove)
    batch_size = image_tensor.size(0)
    coloc_maps = list()

    for i in np.arange(batch_size):
        coloc = matchmap_generate(image_op[i], caption_op[i])
        coloc = coloc.mean(0)
        mm = coloc.detach().numpy()
        coloc_maps.append(mm)

    return coloc_maps


def gen_coloc_maps_phrase(image_model, caption_model,
                   image_tensor, caption_glove):
    """
    This gen_coloc processes a single glove embedding for a phrase.
    Generate a list of a single match-maps from a batch of data
    :param models and tensors:Caption glove and image tensors required 
    generate a list of co_localization maps.
    :return: A list of co-localization maps. Each map is a numpy ndarray
    """
    image_op = image_model(image_tensor)
    caption_op = caption_model(caption_glove)
    # caption_op = caption_op.mean(1).unsqueeze(1)
    batch_size = image_tensor.size(0)
    coloc_maps = list()

    for i in np.arange(batch_size):
        caption_matrix = caption_op[i]
        caption_emb = caption_matrix.mean(0).unsqueeze(0)
        coloc = matchmap_generate(image_op[i], caption_emb)
        coloc = coloc.squeeze(0)
        mm = coloc.detach().numpy()
        coloc_maps.append(mm)

    return coloc_maps


def rgb2gray_genome(rgb):
    """
    Convert color numpy image to grayscale
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def tensor2img_genome(tensor_image):
    """
    Convert each tensor image to numpy image
    :return image: Dictionary of two elements
    :return image['col']: Numpy 3 channel color image
    :return image['bw']: Numpy single channel BW image
    """
    img = tensor_image.permute(1, 2, 0)
    color_img = img.numpy()
    bw_img = rgb2gray_genome(color_img)
    color_img = (color_img - np.amin(color_img)) / np.ptp(color_img)
    image = {"color": color_img, "bw": bw_img}

    return image


def bbox_processor(bbox, image_size):
    """
    Adjust bounding box to have compatible dimensions
    """
    x1 = bbox[0]
    x2 = bbox[2] + bbox[0]

    y1 = bbox[1]
    y2 = bbox[1] + bbox[3]

    im_ht = image_size[0]
    im_wt = image_size[1]

    new_box = list()
    hf = 224/im_ht
    wf = 224/im_wt

    new_box.append(x1 * wf)
    new_box.append(y1 * hf)
    new_box.append(x2 * wf)
    new_box.append(y2 * hf)

    return new_box


def genome_coloc_processor(coloc_map):
    """
    Process a list of coloc maps
    :param coloc_map: List of numpy matrices. Clipped coloc map.
    :return mask_list: numpy stack of resized masks to be visualized

    mask_list = list()
    for frame in coloc_map:
        mask = cv2.resize(frame, dsize=(224, 224))
        mask_list.append(mask)
    mask_list = np.stack(mask_list, axis=0)
    return mask_list
    """
    mask = cv2.resize(coloc_map, dsize=(224,224))
    return mask


def fetch_data_genome(index, coloc_maps, image_tensor, ann_ids):
    """
    Parse all necessary data from a batch based on index into a dictionary
    :return element: Dictionary containing three items:
    1. image tensor at index-th index
    2. caption/Caption ID depending on dataset being used
    3. coloc_map: untrimmed co_loc map at index position.
    """
    element = dict()
    element['image'] = image_tensor[index]
    element['caption_id'] = ann_ids[index]
    element['coloc_map'] = coloc_maps[index]

    return element


def genome_element_processor(element, image_data, caption_data):
    """
    Element should have:
    image, caption string, colocalization map, bounding box
    """
    # Fetch caption_id, image_id and string phrase
    caption_id = element['caption_id']
    image_id = str(caption_data[caption_id]['image_id'])
    caption_string = caption_data[caption_id]['phrase']
    
    # Process Bboxes
    bboxes = caption_data[caption_id]['bbox']
    image_size = [image_data[image_id]['height'], image_data[image_id]['width']]
    new_bboxes = bbox_processor(bboxes, image_size)

    # Process image
    new_image = tensor2img_genome(element['image'])

    # Process co-localization map
    new_coloc = genome_coloc_processor(element['coloc_map'])

    processed_elem = dict()
    processed_elem['caption'] = caption_string
    processed_elem['caption_id'] = caption_id
    processed_elem['image'] = new_image
    processed_elem['bboxes'] = new_bboxes
    processed_elem['coloc_map'] = new_coloc

    return processed_elem


def find_mask_max_genome(element):
    """
    Find the coordinates of the maximum point in each mask
    :param mask_list: list of co localization masks
    :return coords_max: list of tuples. Each tuple is max_coordinate
    """
    mask = element['coloc_map']
    mask = element['coloc_map']
    ind = np.unravel_index(np.argmax(mask, axis=None), mask.shape)
    return list(ind)


def hit_score_genome(element):
    """
    Return 1/0 depending on whether max point is in bounding box
    """
    mask_pt = find_mask_max_genome(element)
    coord = element['bboxes']
    return int(coord[0] <= mask_pt[0] <= coord[2]) and (coord[1] <= mask_pt[1] <= coord[3])


def mask_viz_genome(element, save_flag=False, save_name=''):
    """
    Visualize attention masks over image
    """
    fig = plt.figure(figsize=(50,100), facecolor='white')
    caption_phrase = element['caption']
    col_image = element['image']['color']
    bw_image = element['image']['bw']
    mask = cv2.resize(element['coloc_map'], dsize=(224,224))
    box = element['bboxes']

    ax1 = fig.add_subplot(121)
    ax1.imshow(col_image)

    ax2 = fig.add_subplot(122)
    ax2.imshow(bw_image)
    ax2.imshow(mask, cmap='jet', alpha=0.5)
    rect = patches.Rectangle((box[0], box[1]), (box[2]-box[0]), (box[3]-box[1]), linewidth=10, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)

    plt.title(caption_phrase, fontdict={'fontsize': 70})
    plt.axis('off')

    plt.show()

    if save_flag:
        fig.savefig(save_name)


def seg_viz_genome(element, thresh=0.5, save_flag=False, save_name=''):
    """
    Visualize segmentation masks
    """
    fig = plt.figure(figsize=(50,100), facecolor='white')
    caption_phrase = element['caption']
    col_image = element['image']['color']
    # bw_image = element['image']['bw']
    mask = cv2.resize(element['coloc_map'], dsize=(224,224))
    mask2 = np.where((mask<thresh*np.mean(mask)), 0, 1).astype('uint8')
    img = col_image * mask2[:,:,np.newaxis]
    box = element['bboxes']

    ax1 = fig.add_subplot(121)
    ax1.imshow(col_image)

    ax2 = fig.add_subplot(122)
    ax2.imshow(img)
    rect = patches.Rectangle((box[0], box[1]), (box[2]-box[0]), (box[3]-box[1]), linewidth=10, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)

    plt.title(caption_phrase, fontdict={'fontsize': 70})
    plt.axis('off')

    plt.show()

    if save_flag:
        fig.savefig(save_name)


