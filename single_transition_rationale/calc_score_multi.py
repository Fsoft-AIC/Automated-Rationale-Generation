"""
This module is used for calculating scores
"""
import os
from shutil import rmtree
import json
import threading
import pandas as pd

from misc.cocoeval import suppress_stdout_stderr, COCOScorer
from pathlib import Path
import argparse



def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def convert_data_to_coco_scorer_format(data):
    """
    This function is used for converting data to coco format
    """
    cap_dict = {}
    pred_dict = {}
    for idx, row in data.iterrows():
        if row['image_id'] in cap_dict:
            cap_dict[row['image_id']] = [{'image_id': row['image_id'],
                                          'caption': row['caption'],
                                          'cap_id': idx}]
        else:
            cap_dict[row['image_id']] = []
            cap_dict[row['image_id']].append({'image_id': row['image_id'],
                                              'caption': row['caption'],
                                              'cap_id': idx})
        if row['image_id'] in pred_dict:
            pred_dict[row['image_id']] = [{'image_id': row['image_id'],
                                           'caption':row['prediction']}]
        else:
            pred_dict[row['image_id']] = []
            pred_dict[row['image_id']].append({'image_id': row['image_id'],
                                              'caption': row['prediction']})

    return cap_dict, pred_dict


# valid score, save
def calc_scores(scorer, cap_dict, pred_dict):
    """
    Calculate scores
    """
    results = []
    with suppress_stdout_stderr():
        valid_score = scorer.score(cap_dict, pred_dict, pred_dict.keys())
    results.append(valid_score)
    return results


def thread_main(json_folder, files_name, mode, files, final_results):
    """
    Single thread of main
    """
    result = {"mode": mode, "data": []}
    for file in files:
        scorer = COCOScorer()
        file_name = os.path.join(json_folder,
                                 files_name + "_ckpt-" + str(file) + "_" + mode + ".json")
        with open(file_name, encoding="UTF-8") as train_file:
            dict_train = json.load(train_file)

        train = pd.json_normalize(dict_train, max_level=1, record_path="data")
        train = train.reset_index()
        train.head()
        train = train[['index', 'true_caption', 'predicted_caption']]
        train = train.rename(columns={'index': 'image_id', 'predicted_caption': 'prediction',
                                      'true_caption': 'caption'})
        cap_dict, pred_dict = convert_data_to_coco_scorer_format(train)
        scores = calc_scores(scorer, cap_dict, pred_dict)
        result["data"].append({"epoch": file*10, "file_name": file_name, "scores": scores[0]})
    with open(final_results, "+w", encoding="UTF-8") as file:
        json.dump(result, file)
    print(f"Done file {final_results}")


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", dest='mode', type=str,
                        default="test", help="train, test, eval")
    parser.add_argument("--json_folder", dest='json_folder', type=str,
                        default="result/captions/pong", help="Caption directory")
    parser.add_argument("--checkpoints", dest='checkpoints', type=int,
                        required=True, help="Checkpoint to calculate scores")
    parser.add_argument("--files_name", dest="files_name", type=str,
                        default="caption_result_pong",
                        help="Caption file name")
    parser.add_argument("--output_folder", dest="output_folder", type=str,
                        default="result/scores/pong",
                        help="Output scores to this folder")
    args = parser.parse_args()
    mode = args.mode
    json_folder = os.path.join(args.json_folder, mode)
    files_name = args.files_name
    output_folder = args.output_folder
    checkpoints = [args.checkpoints]
    files_splited = list(split(checkpoints, 5))
    try:
        rmtree(os.path.join(output_folder, mode))
    except Exception:
        pass
    Path(os.path.join(output_folder, mode)).mkdir(parents=True, exist_ok=True)
    total_json = os.path.join(output_folder, mode,
                              "scores_all_" + mode + ".json")
    scores_jsons = []
    for i in range(5):
        scores_jsons.append(os.path.join(output_folder, mode,
                            "scores_" + str(i) + "_" + mode + ".json"))
    t_1 = threading.Thread(target=thread_main, args=(json_folder, files_name,
                                                     mode, files_splited[0], scores_jsons[0]))
    t_2 = threading.Thread(target=thread_main, args=(json_folder, files_name,
                                                     mode, files_splited[1], scores_jsons[1]))
    t_3 = threading.Thread(target=thread_main, args=(json_folder, files_name,
                                                     mode, files_splited[2], scores_jsons[2]))
    t_4 = threading.Thread(target=thread_main, args=(json_folder, files_name,
                                                     mode, files_splited[3], scores_jsons[3]))
    t_5 = threading.Thread(target=thread_main, args=(json_folder, files_name,
                                                     mode, files_splited[4], scores_jsons[4]))

    t_1.start()
    t_2.start()
    t_3.start()
    t_4.start()
    t_5.start()

    t_1.join()
    t_2.join()
    t_3.join()
    t_4.join()
    t_5.join()

    all_scores = {"mode": mode, "data": []}
    max_Bleu_4 = 0
    best_epoch = -1
    for scores_json in scores_jsons:
        with open(scores_json, encoding="UTF-8") as scores_file:
            dict_scores = json.load(scores_file)
        for data in dict_scores["data"]:
            if data["scores"]["Bleu_4"] > max_Bleu_4:
                best_epoch = data["epoch"]
                max_Bleu_4 = data["scores"]["Bleu_4"]
            all_scores["data"].append(data)
    all_scores["summary"] = {"max_Bleu_4": max_Bleu_4, "best_epoch": best_epoch}
    with open(total_json, "+w", encoding="UTF-8") as file:
        json.dump(all_scores, file)


if __name__ == "__main__":
    main()
