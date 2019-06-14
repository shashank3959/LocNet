import nltk
import os
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import json


class VisualGenome(data.Dataset):

    def __init__(self, transform, mode, batch_size,
				annotations_file, img_folder, vocab_glove_file,
				start_word='<start>', end_word='<end>', unk_word='<unk>',
				pad_caption=True, pad_limit=20):

        self.mode = mode
        self.img_folder = img_folder
        self.vocab_glove = json.load(open(vocab_glove_file, encoding='utf-8', mode='r'))
        self.transform = transform
        self.batch_size = batch_size
        self.pad_caption = pad_caption
        self.pad_limit = pad_limit
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word

		# All Phrase descriptions for COCO Images
        self.annotations = json.load(open(annotations_file, encoding='utf-8', mode='r'))
		# All phrase IDs
        self.ids = list(self.annotations.keys())
        all_tokenized_captions = list()
        for caption_id in self.annotations:
            all_tokenized_captions.append(self.annotations[caption_id]['tok_phrase'])
        self.caption_lengths = [len(caption) for caption in all_tokenized_captions]

    def __getitem__(self, index):

        if self.mode in ['train', 'test']:
            ann_id = self.ids[index]
            image_id = self.annotations[ann_id]['image_id']
            image_file = str(image_id)+'.jpg'
            image = Image.open(os.path.join(self.img_folder, image_file)).convert("RGB")
            image = self.transform(image)
            caption_tokens = self.annotations[ann_id]['tok_phrase']
            caption = self.process_captions(caption_tokens)
            caption_gloves = torch.Tensor([self.token_glove_generator(item) for item in caption])

            return image, caption_gloves, ann_id

    def process_captions(self, list_of_items):
        """
        Creates a padded list of items based on requirements
        :param list_of_items: List of words
        :return: padded list of captions.
        """
        processed_list = list()
        processed_list.append(self.start_word)
        processed_list.extend(list_of_items)
        processed_list.append(self.end_word)
        if self.pad_caption:
            processed_list.extend([self.end_word]*(self.pad_limit - len(list_of_items)))

        return processed_list

    def token_glove_generator(self, word):
        """
        Generates a 300-d embedding for a word
        :param word: a string representing a word
        """
        if word not in self.vocab_glove.keys():
            return self.vocab_glove['<unk>']

        return self.vocab_glove[word]

    def get_indices(self):
        if self.pad_caption:
            all_indices = np.where([self.caption_lengths[i] <= self.pad_limit for i in np.arange(len(self.caption_lengths))])[0]
        else:
            sel_length = np.random.choice(self.caption_lengths)
            all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]

        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
    	return len(self.ids)
