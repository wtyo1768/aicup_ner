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
    --lattice 1 \
    --only_bert 1 \
    --weight_decay 0.01 \
    --after_bert 'lstm' \
    --warmup 0.3 \
    --optim 'adam' \
    --fix_bert_epoch 0 \
    --epoch 5 \
    --batch 16 \
    --status 'train' \
    --lexicon_name 'lk' \
    --bigram_min_freq 1 \
    --embed_lr_rate 1.3 \
    --fold 3 \
    --use_abs_pos false \
    --use_rel_pos true \
    --do_pred 0

# bagging with no bigram
# TODO 等價字替換 nlpcda