cd ./V1


python3 flat_main.py \
    --dataset 'aicup' \
    --number_normalized 3 \
    --cv true \
    --use_pos_tag 1 \
    --model_type 'many' \
    --use_bigram 0 \
    --layer 1 \
    --use_bert 1 \
    --only_bert 0 \
    --weight_decay 0.03 \
    --after_bert 'mlp' \
    --warmup 0.1 \
    --optim 'adam' \
    --fix_bert_epoch 0 \
    --epoch 1 \
    --batch 16 \
    --status 'train' \
    --lexicon_name 'lk' \
    --bigram_min_freq 1 \
    --embed_lr_rate 1 \
    --fold 3 \
    --use_abs_pos false \
    --use_rel_pos true \
    --do_pred 1

# bagging with no bigram
# TODO 等價字替換 nlpcda