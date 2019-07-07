import os


def move_to_dest(token, f_type, fold):
    mod_token = token + file_types[f_type][0]
    f_name = os.path.join(file_types[f_type][1], mod_token)
    new_fname = os.path.join(file_types[f_type][1], fold, mod_token)
    os.rename(f_name, new_fname)


if __name__ == '__main__':

    fold_files = {"train": "train.txt",
                  "val": "val.txt",
                  "test": "test.txt"}

    img_folder = 'flickr30k-images'
    sent_folder = os.path.join('annotations_flickr', 'Sentences')
    ann_folder = os.path.join('annotations_flickr', 'Annotations')

    file_types = {"img": ['.jpg', img_folder],
                  "sent": ['.txt', sent_folder],
                  "ann": ['.xml', ann_folder]}

    for fold in fold_files:

        os.makedirs(os.path.join(img_folder, fold), exist_ok=True)
        os.makedirs(os.path.join(sent_folder, fold), exist_ok=True)
        os.makedirs(os.path.join(ann_folder, fold), exist_ok=True)

        print("Now doing fold: ", fold)
        with open(fold_files[fold], encoding='utf-8', mode='r') as f:
            count = 0
            for file_name in f:
                if file_name[-1] == '\n':
                    move_to_dest(file_name[:-1], "img", fold)
                    move_to_dest(file_name[:-1], "sent", fold)
                    move_to_dest(file_name[:-1], "ann", fold)

                    count += 1

            print("Done with fold : {fold},\n number of files: {count}".format(fold=fold, count=count))
