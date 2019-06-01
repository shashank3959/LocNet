from .visualize_utils import *

from torchvision import transforms
import json
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from tqdm import tqdm

# from steps.utils import *

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])


class FlickrViz():

    def __init__(self, batch_size, parse_mode, model_path='saved_models/checkpoint.pth.tar', 
                 mode='test', transform=transform, eval_mode=False):
        """
        If eval_mode is true, evaluate localization score for entities present in the dataset.
        Otherwise, load models, batch-size data, compute colocalization maps. 
        """
        self.eval_mode = eval_mode
        self.batch_size = batch_size
        self.parse_mode = parse_mode
        self.json_path = 'data/flickr_30kentities/annotations_flickr/Sentences/test/data.json'
        self.data = json.load(open(self.json_path, encoding='utf-8', mode='r'))
        self.model_path = model_path
        self.mode = mode
        self.transform = transform
        self.image_model, self.caption_model = get_models(self.model_path)
        
        if self.eval_mode:
            loader = flickr_load_data(1, self.parse_mode, self.transform,
                                      self.mode, self.eval_mode)
            self.dataset = loader.dataset

        else:
            self.image_tensor, self.caption_glove, self.ids = flickr_load_data(self.batch_size,
                                                                           self.parse_mode,
                                                                           self.transform)
            self.coloc_maps = gen_coloc_maps(image_model, caption_model,
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
        raw_element = fetch_data(index,self.coloc_maps, self.image_tensor, self.ids)
        self.element = flickr_element_processor(raw_element, self.parse_mode, self.data)

        return self.element

    def __call__(self, save_flag=False, seg_flag=False, thresh=0.5):
        """
        When called, the instance will generate heat maps for the given image and entities.
        :param save_flag: to save results as jpg files or not
        :param seg_flag: to show localization as segmentation masks or heat maps
        :param thresh: threshold for segmentation
        """
        element = self.element
        color_img = element['image']['color']
        color_img = (color_img - np.amin(color_img)) / np.ptp(color_img)
        bw_img = element['image']['bw']

        mask_list = element['coloc_map']
        caption = phrase_detokenizer(element['caption'])

        plt.imshow(color_img)
        plt.title("Original Image")
        plt.axis("off")
        plt.show()

        save_name_results = ''

        if save_flag:
            save_name_original = element['name'] + '_original.png'
            save_name_results = element['name'] + '_results.png'
            plt.imsave(save_name_original, color_img)

        if seg_flag:
            seg_viz(mask_list, caption, color_img, thresh, save_flag, save_name_results)

        else:
            mask_viz(mask_list, caption, bw_img, save_flag, save_name_results)

    def loc_eval(self, last):
        """
        When in eval mode, this will load entire dataset and iteratively find
        localization score for each image first and then average it to find 
        localization score for the entire dataset. Only works when eval_mode is True.
        :param last: number of images to evaluate. For full dataset, use len(data_loader.dataset)
        :return score_list: last - length list of all scores
        :return mean(score_list): mean localization score for dataset. 
        """
        score_list = list()
        for index in np.arange(last):
            image_tensor, caption_glove, caption, cap_id = self.dataset[index]
            image_tensor = image_tensor.unsqueeze(0)
            caption_glove = caption_glove.unsqueeze(0)
            co_loc_map = gen_coloc_maps(self.image_model, self.caption_model,
                                        image_tensor, caption_glove)
            element = fetch_data(0, co_loc_map, image_tensor, cap_id)
            element = flickr_element_processor(element, self.parse_mode, self.data)
            score = element_score(element)
            score_list.append(score)
            print(score, mean(score_list))

        return mean(score_list), score_list



