import argparse
from visualizations import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--model_path', default=0, type=int)
parser.add_argument('--parse_mode', defaut='phrase', type=str)

dict_models = {none_model_path:'',
			   model_path_30_epochs:'saved_models/flickr_30epochs.tar',
			   model_path_100_epochs:'saved_models/flickr_100epochs.tar'}

model_paths = list(dict_models.values())

args = parser.parse_args()

model_path = model_paths[args.model_path]

flickr_processor = FlickrViz(batch_size = 1, parse_mode=args.parse_mode, model_path=, eval_mode=True)
length_dataset = len(flickr_processor.dataset)

score, score_list = flickr_processor.loc_eval(length_dataset)

print(score)