from __future__ import annotations
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from models import CNN_Encoder, RNN_Decoder
from tokenizer import tokenize
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def one_hot(_action):
    out = np.zeros(6, dtype=np.float32)
    out[int(_action)] = 1.0
    return out


# Load the numpy files and actions
def map_func(_input, cap):
    feat_dir = args.feat_dir
    _img_tensor_0 = np.load(os.path.join(feat_dir, _input[0].decode('utf-8')+'.npy'))
    _img_tensor_1 = np.load(os.path.join(feat_dir, _input[1].decode('utf-8')+'.npy'))
    _action = one_hot(_input[2])
    return _img_tensor_0, _img_tensor_1, _action, cap


# Loss function
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


# Train step
@tf.function
def train_step(_img_tensor_0, _img_tensor_1, _action, _target):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=_target.shape[0])

    dec_input = tf.expand_dims([word_to_index('<start>')] * _target.shape[0], 1)

    with tf.GradientTape() as tape:
        feature_0, feature_1 = encoder(_img_tensor_0, _img_tensor_1)

        for i in range(1, _target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _, _ = decoder(dec_input, feature_0, feature_1, _action, hidden)
            loss += loss_function(_target[:, i], predictions)

            # using teacher /forcing
            dec_input = tf.expand_dims(_target[:, i], 1)

    _total_loss = (loss / int(_target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, _total_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dir", dest="feat_dir", type=str,
                        default="dataset/feats/inceptionv3/pong_single_24_3")
    parser.add_argument("--json_file", dest='json_file', type=str,
                        default="dataset/datainfo/24_3/datainfo_pong.json",
                        help="Splitted json file location")
    parser.add_argument("--checkpoint_dir", dest='checkpoint_dir', type=str,
                        default="./checkpoints/pong", help="Checkpoint directory")
    parser.add_argument("--max_length", dest='max_length', type=int,
                        default=40, help='Max length of caption')
    parser.add_argument("--vocab_size", dest='vocab_size', type=int,
                        default=5000, help='Max vocab size')
    parser.add_argument("--batch_size", dest='batch_size', type=int,
                        default=100, help='Batch size')
    parser.add_argument("--embedding_dim", dest='embedding_dim', type=int,
                        default=512, help='Embedding dimension')
    parser.add_argument("--units", dest='units', type=int,
                        default=512, help='Dimension of feature vectors')
    parser.add_argument("--epoch", dest='epoch', type=int,
                        default=1000, help='Numer of epoch')
    parser.add_argument("--check_point", dest="check_point", type=str,
                        default="None", help="example: ckpt-1")
    parser.add_argument("--loss_array", dest="loss_array", type=str,
                        default="None", help="example: loss_1000_epochs.npy")
    args = parser.parse_args()
    print("Tokenizing ...")
    tokenizer, word_to_index, index_to_word, [img_pairs_train, cap_train], \
        [img_pairs_test, cap_test], [_, _] = tokenize(data_split=args.json_file,
                                                      max_length=args.max_length)
    # Feel free to change these parameters according to your system's configuration
    print("Initializing model ...")
    batch_size = args.batch_size
    BUFFER_SIZE = 1000
    embedding_dim = args.embedding_dim
    units = args.units
    num_steps = len(img_pairs_train) // batch_size
    # Shape of the vector extracted from InceptionV3 is (64, 2048)
    # These two variables represent that vector shape
    features_shape = 2048
    attention_features_shape = 64

    dataset = tf.data.Dataset.from_tensor_slices((img_pairs_train, cap_train))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
            map_func, [item1, item2], [tf.float32, tf.float32, tf.float32, tf.int64]),
            num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle and batch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Initialize model
    encoder = CNN_Encoder(512)
    decoder = RNN_Decoder(embedding_dim, units, tokenizer.vocabulary_size())
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction='none')
    checkpoint_path = args.checkpoint_dir
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=None)

    start_epoch = 0
    if args.check_point == "None":
        if ckpt_manager.latest_checkpoint:
            start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1]) * 10
            # restoring the latest checkpoint in checkpoint_path
            ckpt.restore(ckpt_manager.latest_checkpoint)
    else:
        start_epoch = int(args.check_point.split('-')[-1]) * 10
        ckpt.restore(os.path.join(ckpt_manager.directory, args.check_point))
    if args.loss_array == "None":
        loss_plot = []
    else:
        loss_plot = np.load(os.path.join(args.checkpoint_dir,
                                         args.loss_array)).tolist()[:start_epoch]
    if len(loss_plot) != start_epoch:
        print(f"Warning, \"--loss_array\" not loaded correctly {len(loss_plot)} != {start_epoch}")
    EPOCHS = args.epoch
    print("Training ...")
    pbar = tqdm(range(start_epoch+1, EPOCHS+1))
    for epoch in pbar:
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor_0, img_tensor_1,
             action, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img_tensor_0, img_tensor_1,
                                            action, target)
            total_loss += t_loss

            if batch % 100 == 0:
                average_batch_loss = batch_loss.numpy()/int(target.shape[1])
                # print(f'Epoch {epoch+1} Batch {batch} Loss {average_batch_loss:.4f}')
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)
        pbar.set_postfix({"Epoch": epoch, "Loss": total_loss.numpy()/num_steps})
        if epoch % 1 == 0:
            ckpt_manager.save()
            print(f" Saved checkpoint for epoch {epoch}")
            np.save(os.path.join(args.checkpoint_dir, f"loss_{EPOCHS}_epochs"),
                    np.asarray(loss_plot, dtype=np.float32))
            plt.plot(loss_plot)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Loss Plot')
            plt.savefig(os.path.join(args.checkpoint_dir, "loss_graph.png"))