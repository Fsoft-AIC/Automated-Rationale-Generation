"""
This module is used for evaluate multiple epoch obtainted from the training phase
"""
from __future__ import annotations
from ast import arg
import os
from shutil import rmtree
import json
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from models import CNN_Encoder, RNN_Decoder
from tokenizer import tokenize
from pathlib import Path


def one_hot(action):
    """
    This function is used for action embedding.
    """
    out = np.zeros(6, dtype=np.float32)
    out[int(action)] = 1.0
    return out


class EvaluateClass:
    """
    This class is used for evaluate model on the train/eval set
    """
    def __init__(self, args, word_to_index, index_to_word,
                 encoder, decoder, max_length, attention_features_shape):
        self.args = args
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length
        self.attention_features_shape = attention_features_shape
        self._img_pairs_batches = []



    def load_data(self, _img_pairs_batches):
        
        for _, _img_pairs_batch in enumerate(_img_pairs_batches):
            img_tensor_val_0_batch = []
            img_tensor_val_1_batch = []
            action_batch = []
            for _, img_pair in enumerate(_img_pairs_batch):
                img_tensor_val_0 = np.load(os.path.join(self.args.feature_dir, img_pair[0]+".npy"))
                img_tensor_val_0_batch.append(img_tensor_val_0)
                # img_tensor_val_0 = tf.expand_dims(tf.convert_to_tensor(img_tensor_val_0), 0)
                img_tensor_val_1 = np.load(os.path.join(self.args.feature_dir, img_pair[1]+".npy"))
                img_tensor_val_1_batch.append(img_tensor_val_1)
                # img_tensor_val_1 = tf.expand_dims(tf.convert_to_tensor(img_tensor_val_1), 0)
                action_batch.append(one_hot(img_pair[2]))
            self._img_pairs_batches.append([img_tensor_val_0_batch,
                                            img_tensor_val_1_batch, action_batch])


    def eval(self):
        """
        This function is used for evaluate the model on the train/eval set
        """
        _result_array = []
        for _, _img_pair_batch in enumerate(self._img_pairs_batches):
            _result_array.append(self.eval_loop(_img_pair_batch))
        return _result_array

    def eval_loop(self, _img_pair_batch):
        batch_size = len(_img_pair_batch[0])
        feature_0, feature_1 = self.encoder(tf.convert_to_tensor(_img_pair_batch[0]),
                                            tf.convert_to_tensor(_img_pair_batch[1]))
        hidden = self.decoder.reset_state(batch_size=batch_size)
        dec_input = tf.expand_dims(tf.repeat([self.word_to_index('<start>')],
                                repeats=batch_size), 1)
        _result = [[] for _ in range(batch_size)]
        done_array = np.zeros(batch_size)
        for _ in range(self.max_length):
            predictions, hidden, _, _ = self.decoder(dec_input, feature_0, feature_1,
                                                     tf.convert_to_tensor(_img_pair_batch[2],
                                                                          dtype=tf.float32),
                                                     hidden)
            predicted_id_batch = tf.random.categorical(predictions, 1).numpy()
            predicted_word_batch = [tf.compat.as_text(self.index_to_word(predicted_id[0]).numpy())
                                    for predicted_id in predicted_id_batch]
            for j in range(batch_size):
                if done_array[j] == 0:
                    predicted_word = predicted_word_batch[j]
                    _result[j].append(predicted_word)
                if predicted_word == '<end>':
                    done_array[j] = 1
                    if min(done_array) == 1:
                        return _result
            dec_input = tf.convert_to_tensor(predicted_id_batch)
        return _result


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", dest='mode', type=str,
                        default="test", help="train, test, eval")
    parser.add_argument("--checkpoint_dir", dest='checkpoint_dir', type=str,
                        default="./checkpoints/pong", help="Checkpoint directory")
    parser.add_argument("--checkpoints", dest='checkpoints', type=int,
                        required=True, help="Checkpoint to evaluate")
    parser.add_argument("--feat_dir", dest="feature_dir", type=str,
                        default="dataset/feats/inceptionv3/pong_single",
                        help="Path to feature directory")
    parser.add_argument("--json_file", dest="json_file", type=str,
                        default="dataset/datainfo/datainfo_pong.json",
                        help="Path to splitted datatinfo json file")
    parser.add_argument("--result_json", dest='result_json', type=str,
                        default="result/captions/pong/test/caption_result_pong",
                        help="Generated caption json file")
    parser.add_argument("-vocab_size", dest='vocab_size', type=int,
                        default=5000, help='Maximum vocab size')
    parser.add_argument("--embedding_dim", dest='embedding_dim', type=int,
                        default=512, help='Embedding dimension')
    parser.add_argument("--units", dest='units', type=int,
                        default=512, help='Dimension of feature vectors')
    parser.add_argument("--batch_size", dest='batch_size', type=int,
                        default=165, help='Batch size')
    parser.add_argument("--max_length", dest='max_length', type=int,
                        default=40, help='Max length of caption')
    args = parser.parse_args()
    # checkpoints = range(1, 86, 1)
    checkpoints = [args.checkpoints]
    mode = args.mode
    max_length = args.max_length
    batch_size = args.batch_size
    try:
        rmtree(os.path.split(args.result_json)[0])
    except Exception:
        pass
    # os.mkdir(os.path.split(args.result_json)[0])
    Path(os.path.split(args.result_json)[0]).mkdir(parents=True, exist_ok=True)
    print("Tokenizing...")
    tokenizer, word_to_index, index_to_word, \
        [img_pairs_train, cap_train], [img_pairs_test, cap_test], \
        [img_pairs_eval, cap_eval] = tokenize(data_split=args.json_file,
                                              max_length=max_length)

    # Feel free to change these parameters according to your system's configuration
    embedding_dim = args.embedding_dim
    units = args.units

    # Shape of the vector extracted from InceptionV3 is (64, 2048)
    # These two variables represent that vector shape
    # Initialize model
    print("Initializing model...")
    encoder = CNN_Encoder(512)
    decoder = RNN_Decoder(embedding_dim, units, tokenizer.vocabulary_size())
    optimizer = tf.keras.optimizers.Adam()
    checkpoint_path = args.checkpoint_dir
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)
    evaluator = EvaluateClass(args=args, word_to_index=word_to_index, index_to_word=index_to_word,
                              encoder=encoder, decoder=decoder, max_length=max_length,
                              attention_features_shape=64)
    if mode == "test":
        img_pairs, cap = img_pairs_test, cap_test
    elif mode == "val":
        img_pairs, cap = img_pairs_eval, cap_eval
    elif mode == "train":
        img_pairs, cap = img_pairs_train, cap_train
    img_pairs_batches = [img_pairs[i:i+batch_size]
                            for i in range(0, len(img_pairs), batch_size)]
    cap_batches = [cap[i:i+batch_size] for i in range(0, len(cap), batch_size)]
    real_caption = [[' '.join([tf.compat.as_text(index_to_word(i).numpy())
                                for i in cap_batch[j] if i not in [0]][1:-1])
                    for j in range(len(cap_batch))] for cap_batch in cap_batches]
    print("Loading data ....")
    evaluator.load_data(img_pairs_batches)
    print("Data loaded !!!")
    for checkpoint in tqdm(checkpoints):
        result_batches = []
        result_json = {"data": []}
        result_json["num_data"] = len(img_pairs)
        # print("Loading model ...")
        ckpt.restore(os.path.join(checkpoint_path, "ckpt-" + str(checkpoint))).expect_partial()
        # print("Model loaded !!!")
        # print("Evaluating...")
        result_batches = evaluator.eval()
        for idx_1, result_batch in enumerate(result_batches):
            for idx_2, result in enumerate(result_batch):
                result_json["data"].append({"img_pair": img_pairs_batches[idx_1][idx_2][0:2],
                                            "action": img_pairs_batches[idx_1][idx_2][2],
                                            "split": mode,
                                            "true_caption": real_caption[idx_1][idx_2],
                                            "predicted_caption": " ".join(result[:-1])})
        with open(args.result_json + "_ckpt-" + str(checkpoint) + "_" + mode + ".json",
                  "+w", encoding="UTF-8") as file:
            json.dump(result_json, file)


if __name__ == "__main__":
    main()
