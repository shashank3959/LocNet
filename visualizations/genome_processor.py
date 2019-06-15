from .visualize_utils import *

from torchvision import transforms
import json
import matlplotlib.pyplot as plt
import numpy as np
from statistics import mean
from tqdm import tqdm


transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

class GenomeViz():

    def __init__(self, batch_size, model_path, transform=transform, eval_mode=False):
        """
        If eval_mode is true, compute localization score.
        Otherwise, load models, batch_size data, compute colocalization maps
        """
        self.transform = transform 
        self.eval_mode = eval_mode
        self.batch_size = batch_size
        self.image_data_file = 'data/visual_genome/coco_image_data.json'
        self.annotations_file = 'data/visual_genome/coco_phrase_data.json'

        self.image_data = json.load(open(self.image_data_file, encoding='utf-8', mode='r'))
        self.annotations_data = json.load(open(self.annotations_file, encoding='utf-8', mode='r'))

        self.image_model, self.caption_model = get_models(self.model_path)

        if self.eval_mode:
            loader = genome_load_data(1, transform)
            self.dataset = loader.dataset
        else:
            self.image_tensor, self.caption_glove, self.ann_ids = genome_load_data(self.batch_size,
                                                                                   self.transform)
            self.coloc_maps = gen_coloc_maps(self.image_model, self.caption_model,
                                             self.image_tensor, self.caption_glove)

    def __getitem__(self, index):
        pass

    def loc_eval(self, last):
        pass





