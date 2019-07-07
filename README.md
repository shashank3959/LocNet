# LocNet
Unsupervised Co-localization of Text and Images

![55680967-7038be00-58e6-11e9-9015-8afc9aa5b781](https://user-images.githubusercontent.com/19747416/57676276-27190f80-75f2-11e9-8571-8a50583c8673.png)


## Preprocessing Steps

- Create a vocabulary of the dataset being used
- Generate a dictionary for each word in the vocabulary with its GloVe embedding


## Data Folder Structure

This structure is after the flickr data shuffler script is run. 
```bash
data
├───flickr_30kentities
│   ├───.ipynb_checkpoints
│   ├───annotations_flickr
│   │   ├───Annotations
│   │   │   ├───test
│   │   │   ├───train
│   │   │   └───val
│   │   └───Sentences
│   │       ├───test
│   │       ├───train
│   │       └───val
│   ├───flickr30k-images
│   │   ├───test
│   │   ├───train
│   │   └───val
│   └───__pycache__
└───mscoco
    ├───annotations
    ├───images
    │   ├───train2014
    │   └───val2014
    ├───v2_Annotations_Train_mscoco
    ├───v2_Annotations_Val_mscoco
    ├───v2_Questions_Train_mscoco
    └───v2_Questions_Val_mscoco
```

## Steps

- Run download_data.sh 
- ``` cd 'data/flickr30k_entities'```
- ``` python flickr_data_shuffler.py```
- ``` python flickr_vocab_gen.py``` (Before running this, make sure you have glove.6B.300d.txt from http://nlp.stanford.edu/data/glove.6B.zip in the data/flickr_30kentities folder)
- ``` python flickr_caption_parser.py```
    - This can be used with required arguments. 
    - mk: decide to create a data.json file out of the parsed captions. If not 'make' then script just prints the json data.
    - 'fold': decide which fold of data to operate on. Default is 'train'.
- ``` cd ../..```
- ``` python .\main.py ``` with necessary args
