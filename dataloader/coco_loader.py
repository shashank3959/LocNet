"""Create the CoCoDataset and a DataLoader for it."""
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


class CoCoDataset(data.Dataset):

    def __init__(self, transform, mode, batch_size, annotations_file,
                 img_folder, vocab_glove_file, start_word="<start>",
                 end_word="<end>", unk_word="<unk>",pad_caption=True,
                 pad_limit=20,fetch_mode='default', data_mode='default',
                 disp_mode='default', test_size=1000):
        self.test_size = test_size
        self.mode = mode
        self.disp_mode = disp_mode
        self.data_mode = data_mode
        self.fetch_mode = fetch_mode
        self.transform = transform
        self.batch_size = batch_size
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.img_folder = img_folder
        self.pad_caption = pad_caption
        self.pad_limit = pad_limit


        # Vanilla case. __getitem__ uses an index from ids [caption IDs list]
        if self.mode in ["train", "val"] and self.disp_mode == 'default':
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())  # Caption IDs
            # print("Obtaining caption lengths...")
            # Tokenized captions and length of each caption
            all_tokens = [nltk.tokenize.word_tokenize(
                str(self.coco.anns[self.ids[index]]["caption"]).lower())
                for index in tqdm(np.arange(len(self.ids)))]
            self.caption_lengths = [len(token) for token in all_tokens]

        # Needed for image caption retrieval recall score calculation
        # sample a number of randomly chosen images and their captions
        elif self.mode == "val" and self.disp_mode == "imgcapretrieval":
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())  # Caption IDs
            self.imgIds = self.coco.getImgIds()  # Image IDs
            rng = random.Random(5000)
            # Fetch a set of images and then the corresponding captions
            self.imgIds1k = random.sample(self.imgIds, 10)
            self.capIds5k = self.coco.getAnnIds(imgIds=self.imgIds1k)

        # If in test mode
        else:
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item["file_name"] for item in test_info["images"]]
        self.vocab_glove = json.load(open(vocab_glove_file, "r"))

    def __getitem__(self, index):

        # Obtain image and caption if in training or validation mode
        if self.mode in ["train", "val"] and self.disp_mode == "default":
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]["caption"]
            img_id = self.coco.anns[ann_id]["image_id"]
            path = self.coco.loadImgs(img_id)[0]["file_name"]

            # Convert image to tensor and pre-process using transform
            image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            image = self.transform(image)

            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = list()
            caption.append(self.start_word)
            caption.extend(tokens)
            caption.append(self.end_word)

            if self.pad_caption:
                caption.extend([self.end_word] * (self.pad_limit - len(tokens)))

            caption_gloves = torch.Tensor([self.vocab_glove[word] if word in self.vocab_glove.keys()
                                           else self.vocab_glove["<unk>"] for word in caption])

            if self.fetch_mode == 'retrieval':
                # Return both caption and caption glove list for viz.
                return image, caption_gloves, caption

            return image, caption_gloves


        if self.mode == "val" and self.disp_mode == "imgcapretrieval" and self.data_mode == "imagecaption":
            image_ids = self.imgIds1k

            images = [] # list of all images
            all_gt_captions = [] # list of lists. each list has ground truth captions.
            all_captions_glove = [] # list of glove tensors for each caption
            finalcaptions5k = [] # just a list of all captions
            img_cap_dict = defaultdict(list) # { image_ID1 : [caption1, caption2, ... ], image_ID2 : ...}
            img_cap_len_dict = {}
            count_img = 0

            for image_id in image_ids:
                # append image to list of images
                path2image = self.coco.loadImgs(image_id)[0]["file_name"]
                image = Image.open(os.path.join(self.img_folder, path2image)).convert("RGB")
                image = self.transform(image)
                images.append(image)

                # Get all caption IDs for the image in question.
                caption_ids = self.coco.getAnnIds(imgIds=image_id)
                # Make a list of all ground truth captions for that image ID
                # Append that list to all_gt_captions
                gt_captions = [item['caption'] for item in self.coco.loadAnns(caption_ids)]
                all_gt_captions.append(gt_captions)

                for caption in gt_captions:
                    caption_tokens = nltk.tokenize.word_tokenize(str(caption).lower())
                    # get_indices() doesn't work in this case.
                    if len(caption_tokens) <= self.pad_limit:
                        img_cap_dict[image_id].append(caption)
                        finalcaptions5k.append(caption)
                        caption1 = list()
                        caption1.append(self.start_word)
                        caption1.extend(caption_tokens)
                        caption1.append(self.end_word)
                        if self.pad_caption:
                            caption1.extend([self.end_word] * (self.pad_limit - len(caption_tokens)))
                        caption_gloves = torch.Tensor([self.vocab_glove[word] if word in self.vocab_glove.keys() else
                                                       self.vocab_glove["<unk>"] for word in caption1])
                        all_captions_glove.append(caption_gloves)

                # Attempts to find number of valid captions for each image ??????
                # valid = captions for an image with less than pad_limit words
                img_cap_len_dict[count_img] = len(img_cap_dict[image_id])
                count_img = count_img + 1
            total_caption_gloves = torch.stack(all_captions_glove, dim=0)
            images_tensor = torch.stack(images, dim=0)
            return images_tensor, all_gt_captions, total_caption_gloves, finalcaptions5k, img_cap_dict, img_cap_len_dict


        # Obtain image if in test mode
        else:
            path = self.paths[index]

            # Convert image to tensor and pre-process using transform
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)

            # Return original image and pre-processed image tensor
            return orig_image, image

    def get_indices(self):
        if self.pad_caption:
            all_indices = np.where([self.caption_lengths[i] <= \
                                    self.pad_limit for i in np.arange(len(self.caption_lengths))])[0]
        else:
            sel_length = np.random.choice(self.caption_lengths)
            all_indices = np.where([self.caption_lengths[i] == \
                                    sel_length for i in np.arange(len(self.caption_lengths))])[0]

        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def get_imgindices(self):
        imgindices = self.imgIds1k
        return imgindices

    def get_capindices(self):
        capindices = self.capIds5k
        return capindices

    def __len__(self):
        if self.mode == "train" or self.mode == "val":
            return len(self.ids)
        else:
            return len(self.paths)
