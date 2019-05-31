import numpy as np
import cv2


def rgb2gray(rgb):
    # Convert color numpy image to grayscale
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def tensor2img(tensor_image):
    # Convert each tensor image to numpy image
    img = tensor_image.permute(1, 2, 0)
    color_img = img.numpy()
    bw_img = rgb2gray(color_img)

    return color_img, bw_img

def caption_list_gen(caption):
    """
    Dataloader parses caption as a list of tuples
    Each tuple holds positional words
    This makes it a list of lists of each sentence's tokenized words.
    """
    caption_list = []
    for i in range(len(caption[0])):  # Number of sentences
        new_caption = []  # For each sentence a new list
        for j in range(len(caption)):  # Number of tokens
            new_caption.append(caption[j][i])  # Go to jth tuple, ith list
        caption_list.append(new_caption)  # Generate the respective caption
    return caption_list  # Return tokenized caption

def clip_list(query_list, caption):
    start = caption.index('<end>')
    del [caption[start:]]
    del [query_list[start:]]
    del [caption[0]]
    del [query_list[0]]

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
