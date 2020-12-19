from typing import Dict, List, Tuple
from src.dataset  import encode_data, AICupDataset, id2tag, split_to_sentence
from src.dataset  import romove_redundant_str, generate_type_id
from seqeval.metrics.sequence_labeling import get_entities
import numpy as np
import pandas as pd
import opencc


max_len = 128
output_path = 'output.tsv'


def load_dev(path='/home/dy/flat-chinese-ner/data/test.txt', simplify=True):
    test_data = []
    with open(path, 'r', encoding='utf8') as f:
        file_text=f.read().encode('utf-8').decode('utf-8-sig')
        converter = opencc.OpenCC('t2s.json')
        if simplify:
            file_text = converter.convert(file_text)

        datas=file_text.split('\n\n--------------------\n\n')[:-1]
        for doc in datas:   
            _, doc = doc.split('\n')
            test_data.append(doc)
    return test_data


def count_article_length(docs):
    article_doc_length = []
    for doc in docs:
        article_doc_length.append(len(doc))
    return article_doc_length


def align_predict(preditions, argmax=True):
    if argmax:
        print(preditions.shape)
        pred = np.argmax(preditions, axis=2)
        argmax
    else:
        pred = preditions

    pred_tag = [list(
       map(lambda ele: id2tag[ele] ,line)
    ) for line in pred]

    # doc_result = np.array(split_docs).flatten()
    print('to be refine!')
    # for idx, doc in enumerate(split_docs):
    #     diff = len(pred_tag[idx])-len(doc)
    #     # print(diff)
    #     for _ in range(diff):
    #         pred_tag[idx].pop()

    return pred_tag


def split_to_pred_per_article(pred, articles_wordnum):
    pred = [item for sublist in pred for item in sublist]

    pred_per_article = []
    word_idx = 0
    for word_num in articles_wordnum:
        end_idx = word_idx + word_num
        pred_per_article.append(pred[word_idx:end_idx])
        word_idx = end_idx
    return pred_per_article


def write_result(dev_data, pred_per_article, offset_mapping=None, origin_doc=None, output_path=output_path):
    output="article_id\tstart_position\tend_position\tentity_text\tentity_type\n"

    for article_id, preds_per_article in enumerate(pred_per_article):
        total_entities = get_entities(preds_per_article)
        for entity_type, start_pos, end_pos in total_entities:
            end_pos += 1
            offset = offset_mapping[article_id][start_pos] if offset_mapping else 0
            start_pos += offset
            end_pos += offset
            
            entity = origin_doc[article_id][start_pos:end_pos]
            line = f'{article_id}\t{start_pos}\t{end_pos}\t{entity}\t{entity_type}\n'
            output+=line

    # import opencc
    # converter = opencc.OpenCC('s2t.json')
    # output = converter.convert(output)  
    with open(output_path,'w',encoding='utf-8') as f:
        f.write(output)


def pred_and_write(model, model_type):
    print('start prediction ...')
    ## prepare dataset
    dev_data = load_dev() #回傳一個字串為一個CASE，沒有答案
    origin_data = dev_data.copy()
    offset_mapping = []
    input_id_types = []

    for idx in range(len(dev_data)):
        dev_data[idx], offset_map = romove_redundant_str(dev_data[idx], dev_mode=False)
        offset_mapping.append(offset_map)
        input_id_types.append(generate_type_id(dev_data[idx], offset_map))

    for idx in range(len(dev_data)):
        dev_data[idx] = dev_data[idx].replace('_', '')
        dev_data[idx] = dev_data[idx].replace('*', '')
        dev_data[idx] = dev_data[idx].replace('^', '')
        dev_data[idx] = dev_data[idx].replace('&', '')
        dev_data[idx] = dev_data[idx].replace('~', '')


    split_docs, type_tensor = split_to_sentence(dev_data, input_id_types, max_len)
    dev_tokens = encode_data(
        split_docs, 
        type_tensor, 
        max_len=max_len,
        use_type_id=True,
        pretrained=model_type,
    )
    dev_ds = AICupDataset(dev_tokens)
     
    ## predict 
    preds = model.predict(dev_ds).predictions
    print('writing file... ')
    
    align_preds = align_predict(preds, argmax=False)
    pred_per_article = split_to_pred_per_article(align_preds, count_article_length(dev_data)) 
    write_result(dev_data, pred_per_article, offset_mapping, origin_data)


def convert_pred_and_write(pred, out_path, vocabs):
    pred = [int(ele) for sublist in pred for ele in sublist]
    dev_data = load_dev()
    origin_data = load_dev(simplify=False)

    offset_map = []
    for idx in range(len(dev_data)):
        dev_data[idx], map_arr = romove_redundant_str(dev_data[idx], dev_mode=True)
        offset_map.append(map_arr)

    pred = [vocabs['label'].to_word(ele) for ele in pred]
    pred_per_article = split_to_pred_per_article([pred], count_article_length(dev_data))
    
    with open(out_path, 'wb') as f:
        print(f'writing {out_path}...')    
        np.save(f, np.array(pred))

    


if __name__ == "__main__":

    from transformers import AutoModelForTokenClassification
    from transformers import Trainer, TrainingArguments 
    
    finetuned = '/home/dy/flat-chinese-ner/results/chinese-roberta-wwm-ext/checkpoint-1000'
    model_type = 'hfl/chinese-roberta-wwm-ext'
    model = AutoModelForTokenClassification.from_pretrained(finetuned)
    trainer = Trainer(model=model)

    pred_and_write(trainer, model_type)