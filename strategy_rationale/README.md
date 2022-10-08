# strategy_rationale
Preprocess features
```
python prepro_feats_img_fc.py --input_path data/imgs/pong --json data/json/pong_strat.json --output_dir data/feats/inception_v3/pong_strat --output_dir_feat data/feats/inception_v3/pong_strat_feats --batch_size 2
```
Prepare vocabulary
```
python prepro_vocab.py --input_json data/json/pong_strat.json --info_json data/json/info_pong.json --caption_json data/json/caption_pong.json
```
Training
```
python train.py --input_json data/json/pong_strat.json --info_json data/json/info_pong.json --caption_json data/json/caption_pong.json --feats_dir data/feats/inception_v3/pong_strat --checkpoint_path save/pong
```
Evaluate model
```
python eval_multi.py --recover_opt save/pong/opt_info.json --saved_model save/pong/model --results_path results/pong --epochs 100
```