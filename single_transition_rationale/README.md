# single_transition_rationale
Preprocess features
```
python pre_pro_feats.py --datainfo_splitted dataset/datainfo/datainfo_pong.json --img_path dataset/imgs/pong_single --feat_dir dataset/feats/inceptionv3/pong_single --batch_size 150
```
Training
```
python train.py --feat_dir dataset/feats/inceptionv3/pong_single --json_file dataset/datainfo/datainfo_pong.json --checkpoint_dir ./checkpoints/pong --batch_size 100
```
Generating rationales
```
python eval.py --mode test --checkpoint_dir checkpoints/pong --checkpoints 3 --feat_dir dataset/feats/inceptionv3/pong_single --json_file dataset/datainfo/datainfo_pong.json -- result_json result/captions/pong/test/caption_result_pong --batch_size 165
```
Calculating scores
```
python calc_score_multi.py --mode test --json_folder result/captions/pong --checkpoints 3 --files_name caption_result_pong --output_folder result/scores/pong
```