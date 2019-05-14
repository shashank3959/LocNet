# LocNet
Unsupervised Co-localization of Text and Images

![55680967-7038be00-58e6-11e9-9015-8afc9aa5b781](https://user-images.githubusercontent.com/19747416/57676276-27190f80-75f2-11e9-8571-8a50583c8673.png)


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
