"""
This module is used for tokenizer
"""
from __future__ import annotations
import collections
import json
import numpy as np
import tensorflow as tf


def standardize(inputs):
    """
    We will override the default standardization of TextVectorization to preserve
    "<>" characters, so we preserve the tokens for the <start> and <end>.
    """
    inputs = tf.strings.lower(inputs)
    return tf.strings.regex_replace(inputs,
                                    r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~",
                                    "")


def tokenize(data_split="dataset/datainfo_split.json",
             max_length=48):
    """
    This function is the main tokenizing
    """
    with open(data_split, "+r", encoding="UTF-8") as file:
        data = json.load(file)
    img_ids = [image["id"] for image in data["images"]]
    img_id_to_caption = collections.defaultdict(list)
    img_name_to_id = collections.defaultdict(int)
    for image in data["images"]:
        img_name_to_id[image["imgs_name"][0]] = image["id"]
    for idx in img_ids:
        for annotation in data["annotations"]:
            if annotation["image_id"] == idx:
                caption_str = "<start> " + annotation["caption"] + " <end>"
                img_id_to_caption[idx].append(caption_str)
    img_name_vector = [image["imgs_name"][0] for image in data["images"]]
    img_to_cap_vector_org = collections.defaultdict(list)
    for image in data["images"]:
        img_to_cap_vector_org[image["imgs_name"][0]] = \
            img_id_to_caption[img_name_to_id[image["imgs_name"][0]]]
    captions = []
    for img_name in img_name_vector:
        caption_list = img_to_cap_vector_org[img_name]
        captions.extend(caption_list)
    caption_dataset = tf.data.Dataset.from_tensor_slices(captions)
    # Max word count for a caption.
    max_length = 48
    # Use the top 5000 words for a vocabulary.
    vocabulary_size = 5000
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=vocabulary_size,
        standardize=standardize,
        output_sequence_length=max_length)
    # Learn the vocabulary from the caption data.
    tokenizer.adapt(caption_dataset)
    # Create the tokenized vectors
    cap_vector = caption_dataset.map(lambda x: tokenizer(x))
    # Create mappings for words to indices and indicies to words.
    word_to_index = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary())
    index_to_word = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary(),
        invert=True)
    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(img_name_vector, cap_vector):
        img_to_cap_vector[img].append(cap)
    img_name_train = []
    cap_train = []
    img_name_test = []
    cap_test = []
    img_name_eval = []
    cap_eval = []
    for idx, image in enumerate(data["images"]):
        if image["split"] == "train":
            img_name_train.extend([img_name_vector[idx]])
            cap_train.extend(img_to_cap_vector[img_name_vector[idx]])
        elif image["split"] == "test":
            img_name_test.extend([img_name_vector[idx]])
            cap_test.extend(img_to_cap_vector[img_name_vector[idx]])
        elif image["split"] == "val":
            img_name_eval.extend([img_name_vector[idx]])
            cap_eval.extend(img_to_cap_vector[img_name_vector[idx]])
    img_name_to_pair = collections.defaultdict(list)
    for image in data["images"]:
        img_name_to_pair[image["imgs_name"][0]] = image["imgs_name"]
    img_name_to_action = collections.defaultdict(list)
    for image in data["images"]:
        img_name_to_action[image["imgs_name"][0]] = str(image["action"])
    img_pairs_train = []
    img_pairs_test = []
    img_pairs_eval = []
    for img in img_name_train:
        img_pairs_train.append([*img_name_to_pair[img], img_name_to_action[img]])
    for img in img_name_test:
        img_pairs_test.append([*img_name_to_pair[img], img_name_to_action[img]])
    for img in img_name_eval:
        img_pairs_eval.append([*img_name_to_pair[img], img_name_to_action[img]])
    return tokenizer, word_to_index, index_to_word, [img_pairs_train, cap_train], \
        [img_pairs_test, cap_test], [img_pairs_eval, cap_eval]


def one_hot(action):
    """
    This function is used for generating one-hot vector of action embedding
    """
    out = np.zeros(6)
    out[action] = 1
    return out


if __name__ == "__main__":
    tokenizer_1, word_to_index_1, index_to_word_1, \
        [img_pairs_train_1, cap_train_1], [img_pairs_test_1, cap_test_1], \
        [img_pairs_eval_1, cap_eval_1] = tokenize()
