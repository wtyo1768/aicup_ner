cd ./V1


python3 flat_main.py \
    --dataset 'aicup' \
    --number_normalized 3 \
    --cv true \
    --data_type 'default' \
    --layer 1 \
    --use_bigram 0 \
    --use_pos_tag 0 \
    --model_type 'bert' \
    --lattice 1 \
    --use_bert 1 \
    --only_bert 1 \
    --weight_decay 0.03 \
    --after_bert 'mlp' \
    --warmup 0.1 \
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
    --do_pred 1 \
    --crf_lr 0.1 \
    --post 'n' \
    # --lr 6e-5 \
    # --bert_lr_rate 0.5 \
    # --embed_lr_rate 1

# bagging with no bigram
# TODO 等價字替換 nlpcda
# tune position embedding
# remove noise