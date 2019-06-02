from .visualize_utils import *

from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from tqdm import tqdm

from steps.utils import *

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
class COCOViz():

    def __init__(self, batch_size, model_path='saved_models/checkpoint.pth.tar',
                 mode='val', transform=transform):
        """
        Load models, batch-size data, compute colocalization maps. 
        """

        self.batch_size = batch_size
        self.model_path = model_path
        self.mode = mode
        self.transform = transform

        self.image_model, self.caption_model = get_models(self.model_path)

        self.image_tensor, self.caption_glove, self.caption = coco_load_data(self.batch_size,
                                                                             self.transform,
                                                                             self.mode)

        self.coloc_maps = gen_matchmap(self.image_model, self.caption_model,
                                           self.image_tensor, self.caption_glove)

    def __getitem__(self, index):
        """
        Fetch an index to create a data element from coloc maps, image and caption data
        :param index: index less than the batch_size of the data pooled
        :return element: data structure containing the image, caption and bounding box information
        used to evaluate and visualize the localization tasks being performed by the system.
        """

        element = fetch_data(index, self.coloc_maps, self.image_tensor, self.caption)
        self.element = coco_element_processor(element)

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
        caption = element['caption']
        boxes = list()
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
            seg_viz(mask_list, caption, color_img, boxes, thresh, save_flag, save_name_results)

        else:
            mask_viz(mask_list, caption, bw_img, boxes, save_flag, save_name_results)



        























