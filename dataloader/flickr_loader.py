"""Create the FlickrDataset and a DataLoader for it."""
import nltk
import os
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json
from collections import defaultdict


'''
General Idea:
- First segregate data into divisions
- For train case, return:
    - Image
    - Tokenized Glove embeddings for each word
    - Glove Embedding matrix for each phrase
- For val case, return:
    - Image
    - Tokenized Glove embeddings for each word
    - Glove Embedding matrix for each phrase.
- For test case, return:


To-Do:
1. sample indices
'''


class FlickrDataset(data.Dataset):

    def __init__(self, transform, mode, batch_size, sentences_root, annotations_root,
                 image_root, vocab_glove_file, start_word = '<start>',
                 end_word='<end>', unk_word='<unk>', pad_captions=True,
                 pad_limit=30, disp_mode='default', fold='train'):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.pad_captions = True
        self.pad_limit = pad_limit
        self.disp_mode = disp_mode

        # Assigning proper data based on fold
        assert self.mode in ["train", "val", "test"], "Enter a valid mode to load data"

        self.image_folder = os.path.join(image_root, self.mode)
        self.sentences_folder = os.path.join(sentences_root, self.mode)
        self.annotations_folder = os.path.join(annotations_root, self.mode)






    def __getitem__(self, index):
        pass

    def get_indices(self):
        pass

    def __len__(self):
        return len(os.listdir(self.image_folder))

