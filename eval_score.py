import argparse
from visualizations import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--model_path', default=0, type=str)
parser.add_argument('--parse_mode', default='phrase', type=str)

dict_models = {'none':'',
        'f_30':'saved_models/flickr_30epochs.tar',
        'f_100':'saved_models/flickr_100epochs.tar',
        'c_70': 'saved_models/coco_70epochs.tar',
        'mix': 'saved_models/mix_140epochs.tar'}



args = parser.parse_args()

model_path = dict_models[args.model_path]

flickr_processor = FlickrViz(batch_size = 1, parse_mode=args.parse_mode, model_path=model_path, eval_mode=True)
length_dataset = len(flickr_processor.dataset)

score, score_list = flickr_processor.loc_eval(length_dataset)

print(score)
