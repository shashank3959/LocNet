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
from statistics import mean
from collections import defaultdict

from .flickr30k_entities_utils import *

class FlickrDataset(data.Dataset):

    def __init__(self, transform, mode, batch_size, sentences_file, sentences_root,
                 annotations_root, image_root, vocab_glove_file, start_word='<start>',
                 end_word='<end>', unk_word='<unk>', pad_caption=True,
                 pad_limit=20, parse_mode='phrase'):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.pad_captions = True
        self.pad_limit = pad_limit
        self.parse_mode = parse_mode  # Parsing is phrases or single words
        self.pad_caption = pad_caption  # Sets a limit on caption length
        self.pad_limit = pad_limit  # Limit for length of caption
        self.vocab_glove = json.load(open(vocab_glove_file, 'r'))
        # Assigning proper data based on fold
        self.image_folder = image_root
        self.sentences_folder = sentences_root
        self.sentences_file = sentences_file
        self.annotations_folder = annotations_root
        self.none_word = "<none>"

        # All sentences
        self.sentences = json.load(open(self.sentences_file, 'r'))
        # Caption IDs
        self.ids = list(self.sentences.keys())

        assert self.mode in ["train", "val", "test"], "Enter a valid mode to load data"

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
        # Obtain image and caption if in 'phrase' parse mode
        if self.mode in ['train', 'val', 'test'] and self.parse_mode == 'phrase':
            ann_id = self.ids[index]
            image_file = self.sentences[ann_id]['image_file']
            image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
            image = self.transform(image)

            caption_tokens = self.sentences[ann_id]['parsed_caption']
            # caption_boxes = self.sentences[ann_id]['boxes']

            caption = self.process_captions(caption_tokens, flag='caption')
            # boxes = self.process_captions(caption_boxes, flag='boxes')

            caption_gloves = torch.Tensor([self.token_glove_generator(item) \
                                           for item in caption])

            return image, caption_gloves, caption, ann_id

        elif self.mode in ['train', 'val', 'test'] and self.parse_mode == 'default':
            ann_id = self.ids[index]
            image_file = self.sentences[ann_id]['image_file']
            image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
            image = self.transform(image)

            caption_tokens = self.sentences[ann_id]['tok_sent']
            caption = self.process_captions(caption_tokens, flag='caption')

            caption_gloves = torch.Tensor([self.token_glove_generator(item) \
                                           for item in caption])

            return image, caption_gloves, caption, ann_id


    def process_captions(self, list_of_items, flag):
        """
        Creates a padded list of items based on requirements
        :param list_of_items: List of words or boxes
        :param flag: decides if caption list needs to be padded or boxes list
        :return: padded list of captions or boxes.
        """
        processed_list = list()

        if flag == "caption":
            processed_list.append(self.start_word)
            processed_list.extend(list_of_items)
            processed_list.append(self.end_word)
            if self.pad_caption:
                processed_list.extend([self.end_word] * (self.pad_limit - len(list_of_items)))

        else:
            processed_list.append(self.none_word)
            processed_list.extend(list_of_items)
            processed_list.append(self.none_word)
            if self.pad_caption:
                processed_list.extend([self.none_word] * (self.pad_limit - len(list_of_items)))

        return processed_list

    def get_glove(self, word):
        """
        Generates a 300-dimensional embedding for a word
        :param word: a string representing a word
        :return: a 300-dimensional numpy array for the GloVE Embedding
        """
        if word not in self.vocab_glove.keys():
            return self.vocab_glove['<unk>']
        return self.vocab_glove[word]

    def token_glove_generator(self, token):
        """
        Returns a glove embedding for a single element or a list of elements
        :param token: token is a list or a string
        :return: Average of glove embeddings of all elements in a list or just of the string item
        """
        if type(token) is not list:
            return self.get_glove(token)

        elif len(token) == 1:
            return self.get_glove(token[0])

        else:
            glove_token = []
            for item in token:
                glove_token.append(self.get_glove(item))
            glove_embedding = list(map(mean, zip(*glove_token)))
            return glove_embedding

    def get_indices(self):
        """
        Depending on pad_caption condition choose a list of indices from ids which fit condition
        :return: List of indices for the batch sampler in the dataloader to handle
        """
        if self.pad_caption:
            all_indices = np.where([self.caption_lengths[i] <= \
                                    self.pad_limit for i in np.arange(len(self.caption_lengths))])[0]
        else:
            sel_length = np.random.choice(self.caption_lengths)
            all_indices = np.where([self.caption_lengths[i] == \
                                    sel_length for i in np.arange(len(self.caption_lengths))])[0]

        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        """
        Size of the dataset class
        :return: Number of captions in the dataset fold without get indices.
        """
        return len(self.ids)
