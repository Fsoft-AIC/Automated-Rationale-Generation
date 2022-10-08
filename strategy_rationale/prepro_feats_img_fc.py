import shutil
import subprocess
import glob
from tqdm import tqdm
import numpy as np
import os
import argparse

import torch
from torch import nn
import torch.nn.functional as F
import pretrainedmodels
from pretrainedmodels import utils
import json
import shutil

from pathlib import Path

C, H, W = 3, 224, 224


def extract_frames(video, dst):
    with open(os.devnull, "w") as ffmpeg_log:
        if os.path.exists(dst):
            print(" cleanup: " + dst + "/")
            shutil.rmtree(dst)
        os.makedirs(dst)
        video_to_frames_command = ["ffmpeg",
                                   # (optional) overwrite output file if it exists
                                   '-y',
                                   '-i', video,  # input file
                                   '-vf', "scale=400:300",  # input file
                                   '-qscale:v', "2",  # quality for JPEG
                                   '{0}/%06d.jpg'.format(dst)]
        subprocess.call(video_to_frames_command,
                        stdout=ffmpeg_log, stderr=ffmpeg_log)


def extract_feats(params, model, load_image_fn, batch_size, jsonfile):
    global C, H, W
    model.eval()

    dir_fc = params['output_dir_feat']
    # if not os.path.isdir(dir_fc):
    #     os.mkdir(dir_fc)
    # print("save video feats to %s" % (dir_fc))
    Path(dir_fc).mkdir(parents=True, exist_ok=True)
    existing_features_list = glob.glob(os.path.join(dir_fc, '*.npy'))
    existing_features_list = [feature.split("/")[-1].split(".npy")[0] for feature in existing_features_list]
    image_list = sorted(glob.glob(os.path.join(params['input_path'], '*.png')))
    image_divided_list = list(chunks(image_list, batch_size))
    di={0:[1,0,0,0,0,0],1:[0,1,0,0,0,0], 2:[0,0,1,0,0,0], 3:[0,0,0,1,0,0], 4:[0,0,0,0,1,0], 5:[0,0,0,0,0,1]}
    f=open(jsonfile)

    data= json.load(f)
    obs_act_dict={}
    for strategy in data["videos"]:
        for obs, action in zip(strategy["list_obs"],strategy["list_actions"]):
            obs_act_dict[obs]=di[action]
            
    for index, image_list in enumerate(tqdm(image_divided_list)):
        
        images = torch.zeros((len(image_list), C, H, W))
        image_id = [None] * len(image_list)
        for index, img_loc in enumerate(image_list):
            img = load_image_fn(img_loc)
            image_id[index] = Path(img_loc).name.split(".png")[0]
            images[index] = img
        with torch.no_grad():
            fc_feats = model(images.cpu()).squeeze()
        img_feats = fc_feats.cpu().numpy()
        # Save the inception features
        for i in range(len(image_id)):
            action=obs_act_dict[image_id[i]]
            outfile = os.path.join(dir_fc, image_id[i] + '.npy')
            output=np.concatenate([img_feats[i, :],action])
            np.save(outfile, output)
        
    dir_fc_combined = params['output_dir']
    Path(dir_fc_combined).mkdir(parents=True, exist_ok=True)
        # image_id[i] -> store -> add current videos_id
    for item  in data['videos']:
        feat_strategy=[]
        video_id=item['video_id']
        for obs in item['list_obs']:
            
            output_frame_i=np.load(os.path.join(dir_fc,str(obs) +'.npy'))
            # feat_strategy=np.vstack([feat_strategy, output_frame_i])
            feat_strategy.append(output_frame_i)
        
        feat_strategy = np.stack(feat_strategy, axis=0)
        outfile=os.path.join(dir_fc_combined,video_id +'.npy')
        np.save(outfile, feat_strategy)
    shutil.rmtree(dir_fc)

def chunks(lst, batch_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest='gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    # strategy feat
    parser.add_argument("--output_dir", dest='output_dir', type=str,
                        default='data/feats/inception_v3/pong_strat', help='directory to store features')
    # single image feat
    parser.add_argument("--output_dir_feat", dest='output_dir_feat', type=str,
                        default='data/feats/inception_v3/pong_strat_feats', help='directory to store single frame features')
    parser.add_argument("--input_path", dest='input_path', type=str,
                        default='data/imgs/pong', help='path to dataset')
    parser.add_argument("--model", dest="model", type=str, default='inception_v3',
                        help='the CNN model you want to use to extract_feats')
    parser.add_argument("--json", dest="jsonfile", type=str, default='data/json/pong_strat.json',
                        help='Batch size for feature extraction')
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=100,
                        help='Batch size for feature extraction')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    params = vars(args)
    if params['model'] == 'inception_v3':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv3(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    elif params['model'] == 'resnet152':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet152(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    elif params['model'] == 'inception_v4':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv4(
            num_classes=1000, pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    else:
        print("doesn't support %s" % (params['model']))
    model.last_linear = utils.Identity()  
    model = nn.DataParallel(model)
    
    model = model.cuda()
    extract_feats(params, model, load_image_fn, params['batch_size'], params['jsonfile'])
