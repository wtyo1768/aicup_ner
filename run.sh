cd ./V1

python3 flat_main.py \
    --dataset 'aicup' \
    --number_normalized 1 \
    --weight_decay 0.03 \
    --after_bert 'mlp' \
    --warmup 0.1 \
    --use_bert 1 \
    --only_bert 0 \
    --optim 'sgd' \
    --fix_bert_epoch 20 \
    --epoch 100 \
    --batch 16 \
    --layer 1 \
    --status 'train' \
    --use_bigram 1 \
    --cv true \
    --model_type 'many' \
    --lexicon_name 'lk' \
    --bigram_min_freq 5