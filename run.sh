cd ./V1

python flat_main.py \
    --dataset 'aicup' \
    --lexicon_name 'yj' \
    --number_normalized 3 \
    --weight_decay 0 \
    --after_bert 'mlp' \
    --warmup 0.1 \
    --use_bert 1 \
    --only_bert 0 \
    --optim 'sgd' \
    --fix_bert_epoch 20 \
    --epoch 20 \
    --batch 16 \
    --layer 1 \
    --status 'train' \
    --use_bigram 1 \
    --cv true \
    --model_type 'no'