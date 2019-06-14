import json
import argparse
import nltk

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--make_file', '--mk', default=False, type=bool,
                    metavar='m', help='whether to create json file out of data')


def gen_image_json(filename, encoding='utf-8', mode='r'):
	image_data = json.load(open(filename, encoding=encoding, mode=mode))
	print("Separating images from COCO and generating subset metadata...")
	coco_images = [item for item in image_data if item['coco_id'] is not None]
	new_image_data = dict()
	for image in coco_images:
		new_image_data[image['image_id']] = {'image_id': image['image_id'],
											'height': image['height'],
											'width': image['width']}

	return new_image_data


def create_boxes(coords, size):
	box_coords = [coords[0], coords[1], (coords[0]+coords[2]), (coords[1] + coords[3])]

	wf = 224/size[0]
	hf = 224/size[1]

	box_coords[0] *= wf
	box_coords[2] *= wf
	box_coords[1] *= wf
	box_coords[3] *= wf

	return box_coords


def gen_region_json(filename, image_ids, encoding='utf-8', mode='r'):
	region_data = json.load(open(filename, encoding=encoding, mode=mode))
	new_phrase_data = dict()
	phrase_counter = 0
	print('Generating data for phrases describing regions in the COCO images...')
	for image in region_data:
		if image['id'] in image_ids:
			for phrase in image['regions']:
				coords = [phrase['x'], phrase['y'], phrase['width'], phrase['height']]
				new_phrase_data[phrase_counter] = {'phrase': phrase['phrase'],
				'tok_phrase': nltk.tokenize.word_tokenize(str(phrase['phrase']).lower()),
				'bbox': coords,
				'image_id': phrase['image_id']}
				phrase_counter += 1
			else:
				continue 

	return new_phrase_data


if __name__ == '__main__':
	args = parser.parse_args()
	if args.make_file:
		print("Going to save files!")
	image_data = gen_image_json('image_data.json')
	image_ids = list(image_data.keys())
	phrase_data = gen_region_json('region_descriptions.json', image_ids)
	if args.make_file:
		print("Creating new JSON files!")
		with open('coco_image_data.json', 'w') as f:
			json.dump(image_data, f)
			print("Created image json file!")
		with open('coco_phrase_data.json', 'w') as f:
			json.dump(phrase_data, f)
			print("Created phrase region json file!")

	else:
		print("Printing section of phrase data!")
		print(list(phrase_data.items())[:5])











