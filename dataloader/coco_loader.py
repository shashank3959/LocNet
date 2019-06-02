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


class COCODataset(data.Dataset):

	def __init__(self, transform, mode, batch_size, annotations_file,
				 img_folder, vocab_glove_file, start_word='<start>',
				 end_word='<end>', unk_word='<unk>', pad_caption=True,
				 pad_limit=20):

		self.mode = mode

		self.img_folder = img_folder
		self.vocab_glove = json.load(open(vocab_glove_file, encoding='utf-8',mode="r"))

		self.transform = transform
		self.batch_size = batch_size
		
		self.pad_caption = pad_caption
		self.pad_limit = pad_limit
		self.start_word = start_word
		self.end_word = end_word
		self.unk_word = unk_word

		if self.mode in ['train', 'val']:
			self.coco = COCO(annotations_file)
			self.ids = list(self.coco.anns.keys())  # Caption IDs

			all_tokens = [nltk.tokenize.word_tokenize(
				str(self.coco.anns[self.ids[index]]["caption"]).lower()) 
				for index in tqdm(np.arange(len(self.ids)))]
			self.caption_lengths = [len(token) for token in all_tokens]

		else:
			test_info = json.loads(open(annotations_file).read())
			self.paths = [item["file_name"] for item in test_info["images"]]


	def __getitem__(self, index):
		assert self.mode in ['train', 'val'], "Attempting to fetch test data."

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

		return image, caption_gloves, caption

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

	def __len__(self):
		if self.mode == "train" or self.mode == "val":
			return len(self.ids)
		else:
			return len(self.paths)

