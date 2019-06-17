import argparse
from visualizations import *
import json

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--model_path', default=0, type=str)
parser.add_argument('--parse_mode', default='phrase', type=str)
parser.add_argument('--dataset', default='genome', type=str)


dict_models = {'none':'',
        'f_30':'saved_models/flickr_30epochs.tar',
        'f_100':'saved_models/flickr_100epochs.tar',
        'c_70': 'saved_models/coco_70epochs.tar',
        'mix': 'saved_models/mix_140epochs.tar',
        'mix2': 'saved_models/coco_genome.tar'}

image_data_file = 'data/visual_genome/coco_image_data.json'
annotations_file = 'data/visual_genome/coco_phrase_data.json'

print("Loading Annotations...")
image_data = json.load(open(image_data_file, encoding='utf-8', mode='r'))
annotations_data = json.load(open(annotations_file, encoding='utf-8', mode='r'))
print("Annotations loaded!")
args = parser.parse_args()

model_path = dict_models[args.model_path]

if args.dataset=='flickr':
    flickr_processor = FlickrViz(batch_size = 1, parse_mode=args.parse_mode, model_path=model_path, eval_mode=True)
    length_dataset = len(flickr_processor.dataset)

    score, score_list = flickr_processor.loc_eval(length_dataset)

    print(score)

else:
    genome_processor = GenomeViz(batch_size=1, model_path=model_path, image_data=image_data, annotations_data=annotations_data, eval_mode=True, parse_mode=args.parse_mode)
    length_dataset = len(genome_processor.dataset)
    score_list, score_mean = genome_processor.loc_eval(length_dataset)