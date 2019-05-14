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


class Flickr30kData(data.Dataset):
    """`Flickr30k Entities <http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/>`_ Dataset.

    Args:
        img_root (string): Root directory where images are downloaded to.
        ann_file (string): Path to annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
    """

    def __init__(self,transform,
                 img_root='data/flickr_30kentities/flickr30k-images/',
                 ann_file='data/flickr_30kentities/results_20130124.token',
                 start_word='<start>',
                 end_word='<end>',
                 unk_word='<unk>',
                 vocab_glove_file="data/flickr_30kentities/vocab_glove_flickr.json",
                 fetch_mode="default",
                 pad_caption=True,
                 pad_limit=20,
                 test_fold="data/flickr_30kentities/test.txt",
                 train_fold="data/flickr_30kentities/train.txt",
                 val_fold="data/flickr_30kentities/val.txt",
                 mode="test",
                 disp_mode="default",
                 num_test=1):

        self.transform = transform
        self.root = img_root
        self.ann_file = os.path.expanduser(ann_file)
        self.vocab_glove = json.load(open(vocab_glove_file, "r"))
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.fetch_mode = fetch_mode
        self.pad_caption = True
        self.pad_limit = pad_limit
        self.test_fold = test_fold
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.mode = mode
        self.disp_mode = disp_mode
        self.num_test = num_test



        if self.mode == "test":
            fold_file = self.test_fold
        elif self.mode == "train":
            fold_file = self.train_fold
        else:
            fold_file = self.val_fold

        f = open(fold_file, encoding='utf-8', mode='r')
        file_names = []

        for file in f:
            # removing last element which is \n
            if file[-1] == "\n":
                file_names.append(file[:-1] + '.jpg')

        # Read annotations and store in a dict
        self.annotations = defaultdict(list)
        with open(self.ann_file, encoding='utf-8') as fh:
            for line in fh:
                img_id, caption = line.strip().split('\t')
                if (len(caption.split(" ")) <= self.pad_limit) and (img_id[:-2] in file_names):
                    self.annotations[img_id[:-2]].append(caption)

        self.ids = list(sorted(self.annotations.keys()))
        print("LEN OF SELF IDS", len(self.ids))

        if self.disp_mode == "imgcapretrieval":
            self.justcaptions21k = []
            with open(self.ann_file, encoding='utf-8') as fh:
                for line in fh:
                    img_id, caption = line.strip().split('\t')
                    if (len(caption.split(" ")) <= self.pad_limit) and (img_id[:-2] in file_names):
                        self.justcaptions21k.append(caption)

        if self.disp_mode == "allretrieval":
            self.annotationsall = defaultdict(list)
            self.justcaptions21k = []
            with open(self.ann_file, encoding='utf-8') as fh:
                for line in fh:
                    img_id, caption = line.strip().split('\t')
                    if (len(caption.split(" ")) <= self.pad_limit) and (img_id[:-2] in file_names):
                        self.justcaptions21k.append(caption)
                    if (len(caption.split(" ")) <= self.pad_limit) and (img_id[:-2] in file_names):
                        self.annotationsall[img_id[:-2]].append(caption)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the image

        Returns:
            tuple: Tuple (image, caption_glove).
            Caption_glove are the glove representation of each word in a
            tokenized caption which has been randomly samples from the
            different captions associated with an image.
        """

        # Captions
        if self.disp_mode == "allretrieval":
            allimages = []
            allgtcaptions = []
            allimageids = []
            all_captions_glove = []
            img_cap_dict = defaultdict(list)
            img_cap_len_dict = {}
            img_imgid_dict = {}
            finalcaptions5k = []
            count_img = 0

            img_id = self.ids

            # for eachimgid in self.ids:
            # for i in range(len(self.ids)):
            for i in range(0, 50):
                eachimgid = self.ids[i]
                allimageids.append(eachimgid)
                filename = os.path.join(self.root, eachimgid)
                image = Image.open(filename).convert('RGB')
                if self.transform is not None:
                    image = self.transform(image)
                target = self.annotations[eachimgid]
                cur_caption = [capt for capt in target]
                allimages.append(image)
                allgtcaptions.append(cur_caption)

                for sent in cur_caption:
                    tokens = nltk.tokenize.word_tokenize(str(sent).lower())
                    if len(tokens) <= self.pad_limit:
                        img_cap_dict[eachimgid].append(sent)
                        finalcaptions5k.append(sent)
                        caption = list()
                        caption.append(self.start_word)
                        caption.extend(tokens)
                        caption.append(self.end_word)
                        if self.pad_caption:
                            caption.extend([self.end_word] * (self.pad_limit - len(tokens)))
                        caption_gloves = torch.Tensor([self.vocab_glove[word] if word in self.vocab_glove.keys() else
                                                       self.vocab_glove["<unk>"] for word in caption])
                        all_captions_glove.append(caption_gloves)

                img_cap_len_dict[count_img] = len(img_cap_dict[eachimgid])
                count_img = count_img + 1
            allimagestensor = torch.stack(allimages, dim=0)
            total_caption_gloves = torch.stack(all_captions_glove, dim=0)
            print("TOTAL CAPTION GLOVES", total_caption_gloves.shape)

        elif self.disp_mode == "default":
            img_id = self.ids[index]
            filename = os.path.join(self.root, img_id)
            image = Image.open(filename).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            target = self.annotations[img_id]
            gtcaption = [capt for capt in target]

            target = self.annotations[img_id]
            # Randomly sample one of the captions for this image
            # :-2 removes the comma and space in the end
            target = random.sample(target, 1)[0]  # [:-2]
            tokens = nltk.tokenize.word_tokenize(str(target).lower())
            caption = list()
            caption.append(self.start_word)
            caption.extend(tokens)
            caption.append(self.end_word)
            if self.pad_caption:
                caption.extend([self.end_word] * (self.pad_limit - len(tokens)))

            caption_gloves = torch.Tensor([self.vocab_glove[word] if word in self.vocab_glove.keys() else
                                           self.vocab_glove["<unk>"] for word in caption])

        # For each word in caption, return its glove-representation
        if self.fetch_mode == 'default' and self.disp_mode == "default":
            return image, caption_gloves
        # Return pre-processed image and caption tensors
        elif self.fetch_mode == 'retrieval' and self.disp_mode == "default":
            return image, caption_gloves, caption

        elif self.fetch_mode == "default" and self.disp_mode == 'allretrieval':
            return allimagestensor, total_caption_gloves, finalcaptions5k, allgtcaptions, img_cap_dict, img_cap_len_dict

    def __len__(self):
        # These are image ids
        return len(self.ids)