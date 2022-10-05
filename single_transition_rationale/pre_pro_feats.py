import numpy as np
import os
import json
import tensorflow as tf
from tqdm import tqdm
import argparse

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datainfo_splitted", dest='datainfo_splitted',
                        type=str, default="dataset/datainfo/24_3/datainfo_pong.json",
                        help='Path to json file with train, test and eval splitted')
    parser.add_argument("--img_path", dest='img_path', type=str,
                        default="dataset/imgs/pong_single_24_3", help='directory to load imgs')
    parser.add_argument("--feat_dir", dest='feat_dir', type=str,
                        default="dataset/feats/inceptionv3/pong_single_24_3", help='Directory to store feature')
    parser.add_argument("--batch_size", dest="batch_size", type=int,
                        default=150, help="Batch size")

    args = parser.parse_args()
    datainfo_split = args.datainfo_splitted
    img_path = args.img_path
    feat_dir = args.feat_dir
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)
        print("Feature dir is created.")
    with open(datainfo_split, "+r", encoding="UTF-8") as file:
        data = json.load(file)
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    img_name_vector = []
    for img in data['images']:
        img_name_vector = [*img_name_vector, *[os.path.join(img_path, img['split'] ,img['imgs_name'][i]) for i in range(2)]]
    image_dataset = tf.data.Dataset.from_tensor_slices(img_name_vector)
    image_dataset = image_dataset.map(load_image,
                                      num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch_size)
    for img, path in tqdm(image_dataset):
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            path_of_feature = os.path.basename(p.numpy().decode("utf-8"))
            np.save(os.path.join(feat_dir, path_of_feature), bf.numpy())
