from transformers import AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import AutoConfig, EvalPrediction
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from seqeval.metrics import classification_report
from src.dataset  import get_dataset, get_label, id2tag, tag2id
from typing import Dict, List, Tuple
from torch import nn
from model import Bert_CRF, Bert_BiLSTM_CRF
import numpy as np
import pandas as pd
import torch
import argparse

#%%
def align_predictions(predictions, label_ids) -> Tuple[List[int], List[int]]:
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list, out_label_list


def compute_metrics(p: EvalPrediction):
    preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
    return {
        # "accuracy": accuracy_score(out_label_list, preds_list),
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }


def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True,)
    parser.add_argument('--classifier', type=str, required=True,)
    parser.add_argument('--max_seq_length', type=int, required=True,)
    parser.add_argument('--num_train_epochs', type=int, required=True,)
    parser.add_argument('--num_train_batchs', type=int, required=True,)
    parser.add_argument('--use_type_id', required=False, action='store_true')
    parser.add_argument('--train_on_all_data', required=False, action='store_true')

    parser.add_argument('--silent', '-s', required=False, action='store_true')
    
    parser.set_defaults(silent=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = process_command()

    train_dataset, val_dataset, test_dataset = get_dataset(
        max_len=args.max_seq_length,
        use_type_id=args.use_type_id,
        pretrained=args.model_name_or_path,
        train_on_all_data=args.train_on_all_data
    )
    unique_tags = get_label()
    label_map = id2tag
    num_labels = len(unique_tags)
    model = {
        'CRF':Bert_CRF,
        'BiLSTM-CRF':Bert_BiLSTM_CRF,
        'Linear':AutoModelForTokenClassification,
    }
    output_dir_prefix = args.model_name_or_path.split('/')[1] if '/' in args.model_name_or_path else args.model_name_or_path
    output_dir = './results/' + output_dir_prefix
    logging_dir = './runs/'  + output_dir_prefix

    training_args = TrainingArguments(
        learning_rate=5e-5,
        do_train=True,
        do_eval=not args.train_on_all_data,
        evaluation_strategy='epoch' if not args.train_on_all_data else 'no',
        output_dir=output_dir,
        logging_dir=logging_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.num_train_batchs,
        per_device_eval_batch_size=args.num_train_batchs,
        warmup_steps=300,
        weight_decay=0.01,
        logging_steps=100,
    )
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id=tag2id,
        num_hidden_layers=10,
        type_vocab_size=2,
        # hidden_dropout_prob=.2
    )
    model = model[args.classifier].from_pretrained(
        args.model_name_or_path,
        config=config,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
#%%
    if not args.train_on_all_data:
        pred = trainer.predict(test_dataset, )
        y_true, y_pred = align_predictions(pred.predictions, pred.label_ids)

        report = classification_report(
            y_true, 
            y_pred, 
            # output_dict=True
        )    
                
        print(report)
    else:
        from predict import pred_and_write
        pred_and_write(trainer, args.model_name_or_path)

#%%

