cd ./V1

python3 flat_main.py \
    --dataset 'aicup' \
    --number_normalized 3 \
    --weight_decay 0.03 \
    --after_bert 'mlp' \
    --warmup 0.1 \
    --use_bert 1 \
    --only_bert 0 \
    --optim 'adam' \
    --fix_bert_epoch 0 \
    --epoch 1 \
    --batch 16 \
    --layer 1 \
    --status 'train' \
    --use_bigram 0 \
    --cv true \
    --model_type 'many' \
    --lexicon_name 'lk' \
    --bigram_min_freq 25 \
    --embed_lr_rate 1.3 \
    --fold 0
