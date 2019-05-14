import os
import torch.utils.data as data

from .coco_loader import CoCoDataset

def get_loader_coco(transform,
               mode="train",
               batch_size=1,
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               num_workers=4,
               cocoapi_loc="",
               vocab_glove_file="data/mscoco/vocab_glove.json",
               fetch_mode='default',
               data_mode='default',
               disp_mode='default',
               test_size=1000,
               pad_caption=True):
    """Return the data loader.
    Parameters:
        transform: Image transform.
        mode: One of "train", "val" or "test".
        batch_size: Batch size (if in testing mode, must have batch_size=1).
        vocab_threshold: Minimum word count threshold.
        vocab_file: File containing the vocabulary.
        start_word: Special word denoting sentence start.
        end_word: Special word denoting sentence end.
        unk_word: Special word denoting unknown words.
        vocab_from_file: If False, create vocab from scratch & override any
                         existing vocab_file. If True, load vocab from
                         existing vocab_file, if it exists.
        vocab_glove_file: This JSON file contains the Glove embeddings for each
                    word in the vocabulary.
        num_workers: Number of subprocesses to use for data loading
        cocoapi_loc: The location of the folder containing the COCO API:
                     https://github.com/cocodataset/cocoapi
        fetch_mode: Indicates mode of retrieving data
    """

    assert mode in ["train", "val", "test"], "mode must be one of 'train', 'val' or 'test'."
    # Based on mode (train, val, test), obtain img_folder and annotations_file
    if mode == "train":
        img_folder = os.path.join(cocoapi_loc, "data/mscoco/images/train2014/")
        annotations_file = os.path.join(cocoapi_loc, "data/mscoco/annotations/captions_train2014.json")
    if mode == "val":
        img_folder = os.path.join(cocoapi_loc, "data/mscoco/images/val2014/")
        annotations_file = os.path.join(cocoapi_loc, "data/mscoco/annotations/captions_val2014.json")
    if mode == "test":
        assert batch_size == 1, "Please change batch_size to 1 if testing your model."
        img_folder = os.path.join(cocoapi_loc, "data/mscoco/images/test2014/")
        annotations_file = os.path.join(cocoapi_loc, "data/mscoco/annotations/image_info_test2014.json")

    # COCO caption dataset
    dataset = CoCoDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_glove_file=vocab_glove_file,
                          img_folder=img_folder,
                          fetch_mode=fetch_mode,
                          data_mode=data_mode,
                          disp_mode=disp_mode,
                          test_size=1000)


    if mode == "train":
        # Randomly sample a caption length, and sample indices with that length.
        indices = dataset.get_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        # data loader for COCO dataset.
        data_loader = data.DataLoader(dataset=dataset,
                                      num_workers=num_workers,
                                      batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                              batch_size=dataset.batch_size,
                                                                              drop_last=False))
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=dataset.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)

    return data_loader
