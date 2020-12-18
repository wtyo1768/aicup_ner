cd ./V1


python3 flat_main.py \
    --dataset 'aicup' \
    --number_normalized 0 \
    --cv true \
    --model_type 'many' \
    --use_bigram 0 \
    --layer 1 \
    --use_bert 1 \
    --only_bert 0 \
    --weight_decay 0.03 \
    --after_bert 'mlp' \
    --warmup 0.05 \
    --optim 'adam' \
    --fix_bert_epoch 0 \
    --epoch 10 \
    --batch 16 \
    --status 'train' \
    --lexicon_name 'lk' \
    --bigram_min_freq 25 \
    --embed_lr_rate 1.2 \
    --fold 0 \
    --use_abs_pos true \
    --use_rel_pos true \
