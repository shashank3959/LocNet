import os
import json
import nltk
import argparse

import sys

path_to_dataloader = '../../'
sys.path.append(path_to_dataloader)

from dataloader.flickr30k_entities_utils import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--folder', '--split', default='train', type=str,
                    metavar='sp', help='which split to operate on')


def main(args):
    path = os.path.join('annotations_flickr', 'Sentences', args.folder)
    data = {}  # This will be converted to json file later
    caption_id = 0
    for item in os.listdir(path):
        image_id = item.replace('.txt', '.jpg')  # Keys in the dictionary.
        all_captions = get_sentence_data(os.path.join(path, item))

        for caption in all_captions:
            data[caption_id] = single_caption_parser(caption, image_id)
            caption_id += 1
        # if image_id not in data:
        #     data[image_id] = []
        # data[image_id].extend(id_processor(all_captions))

    create_json(path, args.folder, data)
    print("JSON File created")


def single_caption_parser(caption, image_file):
    new_caption, phrase_words, phrase_list = phrase_data(caption)

    final_caption = []
    index = 0
    while index < len(new_caption['tok_sent']):
        if not new_caption['tok_sent'][index] in phrase_words:
            final_caption.append([new_caption['tok_sent'][index]])
            index += 1
        else:
            i = index + 1
            phrase = [new_caption['tok_sent'][index]]
            while i < len(new_caption['tok_sent']) and new_caption['tok_sent'][i] in phrase_words:
                phrase.append(new_caption['tok_sent'][i])
                i += 1
                if phrase in phrase_list:
                    break
            final_caption.append(phrase)
            index = i

    new_caption['parsed_caption'] = final_caption
    new_caption['image_file'] = image_file
    return new_caption


def phrase_data(caption):
    # New dictionary created. Doesn't change original data.
    new_caption = {'sentence': caption['sentence'],
                   'tok_sent': nltk.tokenize.word_tokenize(str(caption['sentence']).lower())}
    phrase_words = []  # List of words in phrase
    phrase_list = []
    for phrase in caption['phrases']:
        phrase_words.extend(nltk.tokenize.word_tokenize(str(phrase['phrase']).lower()))
        phrase_list.append(nltk.tokenize.word_tokenize(str(phrase['phrase']).lower()))

    return new_caption, phrase_words, phrase_list


def create_json(path, file_name, data):
    full_path = './' + path + '/' + 'data.json'
    with open(full_path, 'w') as f:
        json.dump(data, f)


# def id_processor(all_captions):
#     id_data = []
#     for caption in all_captions:
#         id_data.append(single_caption_parser(caption))
#     return id_data


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
