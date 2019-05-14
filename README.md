# LocNet
Unsupervised Co-localization of Text and Images

## Preprocessing Steps

- Create a vocabulary of the dataset being used
- Generate a dictionary for each word in the vocabulary with its GloVe embedding


## Data Folder Structure

```bash
├───flickr_30kentities
│   ├───annotations_flickr
│   │   ├───Annotations
│   │   └───Sentences
│   └───flickr30k-images
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

- Explore args in main.py
- run ``` python .\main.py ```
