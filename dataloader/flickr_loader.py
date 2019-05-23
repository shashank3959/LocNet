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

from .flickr30k_entities_utils import *

"""
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
    - Image
    - Tokenized Glove embeddings for each word
    - GLove embedding for each phrase
    - BBOX coordinates for each phrase
"""


class FlickrDataset(data.Dataset):

    def __init__(self, transform, mode, batch_size, sentences_root, annotations_root,
                 image_root, start_word='<start>',
                 end_word='<end>', unk_word='<unk>', pad_captions=True,
                 pad_limit=30, parse_mode='phrase', fold='train'):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.pad_captions = True
        self.pad_limit = pad_limit
        self.parse_mode = parse_mode  # Parsing is phrases or single words

        # Assigning proper data based on fold
        # assert self.mode in ["train", "val", "test"], "Enter a valid mode to load data"

        self.image_folder = os.path.join(image_root, self.mode)
        self.sentences_folder = os.path.join(sentences_root, self.mode)
        self.sentences_file = os.path.join(self.sentences_folder, 'data.json')
        self.annotations_folder = os.path.join(annotations_root, self.mode)


        f = open(self.sentences_file, 'r')
        # All sentences
        self.sentences = json.load(f)
        # Caption IDs
        self.ids = self.sentences.keys()

        if self.parse_mode == 'phrase':  # If parsing is done phrase-wise
            all_tokenized_captions = []
            for caption_id in self.sentences:
                all_tokenized_captions.append(self.sentences[caption_id]['parsed_caption'])
            self.caption_lengths = [len(caption) for caption in all_tokenized_captions]


        else:  # If no parsing exists
            all_tokenized_captions = []
            for caption_id in self.sentences:
                all_tokenized_captions.append(self.sentences[caption_id]['tok_sent'])
            self.caption_lengths = [len(caption) for caption in all_tokenized_captions]

    def __getitem__(self, index):
        pass

    def get_indices(self):
        pass

    def __len__(self):
        return len(self.ids)
