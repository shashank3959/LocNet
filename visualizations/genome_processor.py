from .genome_utils import *

from torchvision import transforms
import json
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from tqdm import tqdm


transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

class GenomeViz():

    def __init__(self, batch_size, model_path, transform=transform, eval_mode=False, parse_mode="matchmap"):
        """
        If eval_mode is true, compute localization score.
        Otherwise, load models, batch_size data, compute colocalization maps
        """
        self.eval_mode = eval_mode
        self.parse_mode = parse_mode
        self.transform = transform 
        
        self.batch_size = batch_size
        self.model_path = model_path
        self.image_data_file = 'data/visual_genome/coco_image_data.json'
        self.annotations_file = 'data/visual_genome/coco_phrase_data.json'

        self.image_data = json.load(open(self.image_data_file, encoding='utf-8', mode='r'))
        self.annotations_data = json.load(open(self.annotations_file, encoding='utf-8', mode='r'))

        self.image_model, self.caption_model = get_models_genome(self.model_path)

        if self.eval_mode:
            loader = genome_load_data(1, transform)
            self.dataset = loader.dataset
        else:
            self.image_tensor, self.caption_glove, self.ann_ids = genome_load_data(self.batch_size,
                                                                                   self.transform)
            if self.parse_mode == 'matchmap':
                self.coloc_maps = gen_coloc_maps_matchmap(self.image_model, self.caption_model,
                                             self.image_tensor, self.caption_glove)
            else:
                self.coloc_maps = gen_coloc_maps_phrase(self.image_model, self.caption_model,
                                                        self.image_tensor, self.caption_glove)



    def __getitem__(self, index):
        """
        Fetch an index to create a data element from coloc maps, image and caption data
        :param index: index less than the batch_size of the data pooled
        :return element: data structure containing the image, caption and bounding box information
        used to evaluate and visualize the localization tasks being performed by the system.
        Can only be used when eval_mode is False.
        """
        assert not self.eval_mode, "Evaluation mode has to be False"
        raw_element = fetch_data_genome(index, self.coloc_maps, self.image_tensor, self.ann_ids)
        self.element = genome_element_processor(raw_element, self.image_data, self.annotations_data)
        self.score = hit_score_genome(self.element)

        return {'element': self.element, 'score':self.score}

    def __call__(self, save_flag=False, seg_flag=False, thresh=0.5, name=''):
        element = self.element
        name = element['caption_id']+'.png'
        if seg_flag:
            seg_viz_genome(element, thresh, save_flag, name)
        else:
            mask_viz_genome(element, save_flag, name)
        print("Score: ",self.score)

    def loc_eval(self, last):
        pass      





