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

parser.add_argument('--make_file', '--mk', default='make', type=str,
                    metavar='m', help='whether to create json file out of data')


def main(args):
    sen_path = os.path.join('annotations_flickr', 'Sentences', args.folder)
    ann_path = os.path.join('annotations_flickr', 'Annotations', args.folder)
    data = {}  # This will be converted to json file later
    caption_id = 0
    for sen_file in os.listdir(sen_path):
        print(os.path.join(sen_path, sen_file))
        image_id = sen_file.replace('.txt', '.jpg')  # Keys in the dictionary.
        ann_file = sen_file.replace('.txt', '.xml')  # Annotation file

        captions = get_sentence_data(os.path.join(sen_path, sen_file))
        annotations = get_annotations(os.path.join(ann_path, ann_file))

        for caption in captions:
            data[caption_id] = single_caption_parser(caption, image_id)
            caption_id += 1
        # if image_id not in data:
        #     data[image_id] = []
        # data[image_id].extend(id_processor(all_captions))

    print("Data dictionary created. Now creating JSON File")
    if args.make_file == 'make':
        create_json(sen_path, data)
        print("JSON File created")
    else:
        print(json.dumps(data, indent=4))
        return


def single_caption_parser(caption, image_file):
    new_caption, phrase_words, phrases = phrase_data(caption)

    final_caption = []
    caption_boxes = []
    index = 0
    while index < len(new_caption['tok_sent']):
        if not new_caption['tok_sent'][index] in phrase_words:
            final_caption.append([new_caption['tok_sent'][index]])
            caption_boxes.append('None')
            index += 1
        else:
            i = index + 1
            phrase = [new_caption['tok_sent'][index]]
            while i < len(new_caption['tok_sent']) and new_caption['tok_sent'][i] in phrase_words:
                phrase.append(new_caption['tok_sent'][i])
                i += 1
                if phrase in list(phrases.values()):
                    break
            final_caption.append(phrase)
            phrase_id = dictionary_search(phrases, phrase)
            caption_boxes.append(phrase_id)
            index = i

    new_caption['parsed_caption'] = final_caption
    new_caption['image_file'] = image_file
    new_caption['box_ids'] = caption_boxes
    return new_caption


def phrase_data(caption):
    # New dictionary created. Doesn't change original data.
    new_caption = {'sentence': caption['sentence'],
                   'tok_sent': nltk.tokenize.word_tokenize(str(caption['sentence']).lower())}
    phrase_words = []  # List of words in phrase
    phrases = {}
    for phrase in caption['phrases']:
        phrase_words.extend(nltk.tokenize.word_tokenize(str(phrase['phrase']).lower()))
        phrases[phrase['phrase_id']] = nltk.tokenize.word_tokenize(str(phrase['phrase']).lower())
        # phrase_list.append(nltk.tokenize.word_tokenize(str(phrase['phrase']).lower()))

    return new_caption, phrase_words, phrases


def dictionary_search(box_dict, query_phrase):
    query_id = 'None'
    for phrase_id, phrase in box_dict.items():
        if phrase == query_phrase:
            query_id = phrase_id

    return query_id


def create_json(path, data):
    full_path = './' + path + '/' + 'data.json'
    with open(full_path, 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
