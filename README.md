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

- If using flickr data, recommend running the flickr data shuffler script first.
- Then run flickr caption parser script that creates a json file, while combining sentence and annotation data.
- Explore args in main.py
- run ``` python .\main.py ```
