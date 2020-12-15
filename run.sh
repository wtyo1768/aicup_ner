cd ./V1

python flat_main.py \
    --dataset 'aicup' \
    --lexicon_name 'yj' \
    --number_normalized 1 \
    --weight_decay 0.01 \
    --after_bert 'mlp' \
    --warmup 0.1 \
    --use_bert 1 \
    --only_bert 0 \
    --optim 'adam' \
    --fix_bert_epoch 5 \
    --epoch 20 \
    --batch 16 \
    --layer 1 \
    --status 'train' \
    --use_bigram 1 \
    --cv true \
    --model_type 'no' \
    --lexicon_name 'lk'