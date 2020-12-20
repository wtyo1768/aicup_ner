cd ./V1


python3 flat_main.py \
    --dataset 'aicup' \
    --number_normalized 3 \
    --cv true \
    --data_type 'number' \
    --layer 1 \
    --use_bigram 0 \
    --use_pos_tag 0 \
    --lattice 1 \
    --use_bert 1 \
    --only_bert 0 \
    --weight_decay 0.03 \
    --after_bert 'mlp' \
    --warmup 0.3 \
    --optim 'adam' \
    --fix_bert_epoch 0 \
    --epoch 15 \
    --batch 16 \
    --status 'train' \
    --lexicon_name 'lk' \
    --bigram_min_freq 1 \
    --embed_lr_rate 1.3 \
    --fold 3 \
    --use_abs_pos true \
    --use_rel_pos false \
    --do_pred 0 \
    --post 'd' 


# bagging with no bigram
# TODO 等價字替換 nlpcda
# tune position embedding
# remove noise