import json
import os
import argparse
from re import T
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader
from models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
from dataloader import VideoDataset, collate_batch
import misc.utils as utils
from misc.cocoeval import suppress_stdout_stderr, COCOScorer
from pathlib import Path
from pandas import json_normalize
from tqdm import tqdm

def convert_data_to_coco_scorer_format(data_frame):
    gts = {}
    for row in zip(data_frame["caption"], data_frame["video_id"]):
        if row[1] in gts:
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': row[0]})
        else:
            gts[row[1]] = []
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': row[0]})
    return gts


def eval_single(model, crit, scorer, dataset, vocab, opt, mode, epoch, max_Bleu_4, best_epoch, score_data):
    caption_file = os.path.join(opt["results_path"],
                 os.path.split(opt["saved_model"])[1] + "_" + str(epoch) + '_' + mode + ".json")
    loader = DataLoader(dataset, batch_size=opt["batch_size"],
                        shuffle=True, collate_fn=collate_batch)
    
    gt_dataframe = json_normalize(
        json.load(open(opt["input_json"]))['sentences'])
    gts = convert_data_to_coco_scorer_format(gt_dataframe)
    for imgs_id in gts.keys():
        gts[imgs_id][0]["caption"] = gts[imgs_id][0]["caption"].strip()
    result = {}
    samples = {}
    # print("Generating predictions for evaluation...")
    for i, data in enumerate(loader):
        # forward the model to get loss
        fc_feats = data['fc_feats'].cuda()
        labels = data['labels'].cuda()
        masks = data['masks'].cuda()
        video_ids = data['video_ids']
      
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq_probs, seq_preds = model(
                fc_feats, mode='inference', opt=opt)

        sents = utils.decode_sequence(vocab, seq_preds)

        for k, sent in enumerate(sents):
            video_id = video_ids[k]
            samples[video_id] = [{'image_id': video_id, 'caption': sent}]

    with suppress_stdout_stderr():
        print("Calculating scores...")
        valid_score = scorer.score(gts, samples, samples.keys())
    score_data.append({"epoch": epoch, "file_name": caption_file, "scores": valid_score})
    result["epoch"] = epoch
    result["file_name"] = caption_file
    result["scores"] = valid_score
    if valid_score["Bleu_4"] > max_Bleu_4["Bleu_4"]:
        best_epoch = epoch
        max_Bleu_4 = valid_score
    for name in samples.keys():
        samples[name][0]["true_caption"] = gts[name][0]["caption"]
    with open(caption_file, 'w') as file:
        json.dump({"predictions": samples, "scores": valid_score},
                  file)
    return max_Bleu_4, best_epoch
    


def main(opt, mode, epochs):
    dataset = VideoDataset(opt, mode)
    opt["vocab_size"] = dataset.get_vocab_size()
    opt["seq_length"] = dataset.max_len
    if opt["model"] == 'S2VTModel':
        model = S2VTModel(opt["vocab_size"], opt["max_len"], opt["dim_hidden"], opt["dim_word"],
                          rnn_dropout_p=opt["rnn_dropout_p"]).cuda()
    elif opt["model"] == "S2VTAttModel":
        encoder = EncoderRNN(opt["dim_vid"],
                             opt["dim_hidden"],
                             bidirectional=bool(opt["bidirectional"]),
                             input_dropout_p=opt["input_dropout_p"],
                             rnn_dropout_p=opt["rnn_dropout_p"])
        decoder = DecoderRNN(opt["vocab_size"], opt["max_len"], opt["dim_hidden"], opt["dim_word"],
                             input_dropout_p=opt["input_dropout_p"],
                             rnn_dropout_p=opt["rnn_dropout_p"], bidirectional=bool(opt["bidirectional"]))
        model = S2VTAttModel(encoder, decoder).cuda()
    all_scores = {"mode": mode, "data": []}
    best_scores = {"Bleu_4": 0}
    best_epoch = -1
    model.eval()
    scorer = COCOScorer()
    for epoch in tqdm(epochs):
        saved_model = opt["saved_model"] + "_" + str(epoch) + ".pth"
        model.load_state_dict(torch.load(saved_model)['model_state_dict'])
        crit = utils.LanguageModelCriterion()
        best_scores, best_epoch = eval_single(model, crit, scorer, dataset, dataset.get_vocab(),
                                             opt, mode, epoch, best_scores, best_epoch, all_scores["data"])
    all_scores["early_stopping"] = {"epoch": best_epoch, "scores": best_scores}
    with open(os.path.join(opt["results_path"], "scores.json"), 'w') as score_file:
        json.dump(all_scores, score_file)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recover_opt', type=str, default="save/pong/opt_info.json",
                        help='recover train opts from saved opt_json')
    parser.add_argument('--saved_model', type=str, default='save/pong/model',
                        help='path to saved model to evaluate')
    parser.add_argument('--results_path', type=str, default='results/pong')
    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu device number')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='minibatch size')
    parser.add_argument('--sample_max', type=int, default=1,
                        help='0/1. whether sample max probs  to get next word in inference stage')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--beam_size', type=int, default=1,
                        help='used when sample_max = 1. Usually 2 or 3 works well.')
    parser.add_argument('--mode', type=str, dest="mode", default="val",
                        help='mode of evaluating')
    parser.add_argument('--epochs', type=int, dest="epochs", default=50,
                        help='epoch to evaluate')
    args = parser.parse_args()
    args = vars((args))
    opt = json.load(open(args["recover_opt"]))
    # epochs = list(range(50, 3001, 50))
    epochs = [args["epochs"]]
    for k, v in args.items():
        opt[k] = v
    opt["results_path"] = os.path.join(opt["results_path"], args["mode"])
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    if not os.path.exists(opt["results_path"]):
        Path(opt["results_path"]).mkdir(parents=True, exist_ok=True)
    main(opt, args["mode"], epochs)
