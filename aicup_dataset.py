from fastNLP_module import StaticEmbedding
from fastNLP.io.loader import ConllLoader
from fastNLP import cache_results
from fastNLP import Vocabulary
from fastNLP import DataSet
from utils import get_bigrams
import os
import sys
from src.dataset import romove_redundant_str, split_to_sentence, all_type, few_type
from src.predict import load_dev
from paths import *
import jieba
import jieba.posseg as pseg
from src.dataset import model_type, HANDLE, model_teamwork, tagging_method, max_len


def get_label_vocab(data_type):
    
    if model_teamwork:
        label = model_type[HANDLE]
    else:
        label = [
            'family', 'education', 'money',
            'med_exam', 'ID', 'contact', 
            'name', 'time', 'location', 'profession'
        ]
    # label = [
    #     'money', 'education', 'name',
    #     'time', 'family', 'med_exam',
    #     'contact', 'location', 'ID', 'profession'
    # ]
    total_label = []
    
    for prefix in tagging_method:
        total_label.extend([prefix+'-' + ele for ele in label])
    total_label.append('O')
    print(total_label)
    label_ds = DataSet({'target':total_label})
    label_vocab = Vocabulary(unknown=None, padding=None)
    label_vocab.from_dataset(label_ds, field_name='target')
    label_vocab.index_dataset(label_ds, field_name='target')
    # label_vocab.add_word_lst(total_label) 
    return label_vocab

@cache_results(_cache_fp='cache/aicupNER_uni+bi', _refresh=True)
def load_aicup_ner(
    path,
    unigram_embedding_path=None,
    bigram_embedding_path=None,
    char_word_dropout=0.01,
    only_train_min_freq=0,
    bigram_min_freq=1,
    data_type='string',
    index_token=True,
    char_min_freq=1,
    cv=False,
    fold=0,
    ):
    vocabs = {}
    embeddings = {}

    train_path = os.path.join(path, f'fold{fold}', f'train/{data_type}')
    dev_path = os.path.join(path, f'fold{fold}', f'dev/{data_type}')
    print('-----------------Dataset---------------------')
    print('loading data from', train_path,'\nand', dev_path)
    loader = ConllLoader(['chars', 'target'])

    train = loader.load(train_path)
    dev = loader.load(dev_path)
 
    ds = {
        'train':train.datasets['train'],
        'dev':dev.datasets['train'],
    }
    ds['aicup_dev'] = get_aicup_devds()
    
    # jieba.enable_paddle()   
    for ds_name in ds.keys():
        ds[ds_name].apply_field(get_bigrams, 'chars', 'bigrams')
        ds[ds_name].add_seq_len('chars', new_field_name='seq_len')

        ds[ds_name].apply_field(get_pos_tag, 'chars', 'pos_tag')

    for k, v in ds.items():
        print('{}:{}'.format(k, len(v)))

    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    pos_vocab = Vocabulary()

    label_vocab = get_label_vocab(data_type)
    pos_vocab.from_dataset(*list(ds.values()), field_name='pos_tag')

    if cv: no_create_entry_ds = [ds['dev'], ds['aicup_dev']]
    else: no_create_entry_ds = [ds['dev'], ds['test'], ds['aicup_dev']]
        
    char_vocab.from_dataset(
        ds['train'],
        field_name='chars',
        no_create_entry_dataset=no_create_entry_ds
    )
    bigram_vocab.from_dataset(
        ds['train'],
        field_name='bigrams',
        no_create_entry_dataset=no_create_entry_ds
    )
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab
    vocabs['pos_tag'] = pos_vocab
    
    if index_token:
        char_vocab.index_dataset(*list(ds.values()), field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(*list(ds.values()),field_name='bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(*list([ds['train'], ds['dev']]), field_name='target', new_field_name='target')
    
    pos_vocab.index_dataset(*list(ds.values()),field_name='pos_tag', new_field_name='pos_tag')
    
    unigram_embedding = StaticEmbedding(
        char_vocab, 
        model_dir_or_name=unigram_embedding_path,                          
        word_dropout=char_word_dropout,                                      
        min_freq=char_min_freq,
        only_train_min_freq=only_train_min_freq,
    )
    bigram_embedding = StaticEmbedding(
        bigram_vocab,
        model_dir_or_name=bigram_embedding_path,
        word_dropout=0.01,
        min_freq=bigram_min_freq,
        only_train_min_freq=only_train_min_freq,
    )
    embeddings['char'] = unigram_embedding
    embeddings['bigram'] = bigram_embedding
    print(ds['train'])
    print(set([ele[0].split('-')[1] if ele[0]!='O' and ele[0][0]!='<' else ele[0] for ele in list(label_vocab)]))
    print('------------------------------------------')
    return ds, vocabs, embeddings


def get_aicup_devds():
    raw_data = load_dev()
    # offset_mapping = []
    for idx in range(len(raw_data)):
        raw_data[idx], _ = romove_redundant_str(raw_data[idx], dev_mode=True)
        # offset_mapping.append(offset_map)

    split_docs, type_tensor = split_to_sentence(raw_data, None, max_len)
    dev_ds = DataSet({'chars':split_docs})  

    return dev_ds


def get_pos_tag(sen):
    sentences = ''.join(sen)
    

    pos = pseg.cut(sentences)

    pos_tag = []
    for words, tag in pos:
        pos_tag += [tag] * len(words)
        # print(words, [tag] * len(words))
    if not len(sen) == len(pos_tag):
        print(sen)
    assert len(sen) == len(pos_tag)
    return pos_tag
    

if __name__ == "__main__":
    
    label = get_label_vocab(data_type='default')
    print(label)
    exit()
    dev_path = os.path.join(aicup_ner_path, f'fold{0}', f'dev/number')
    print('-----------------Dataset---------------------')
    loader = ConllLoader(['chars', 'target'])

    dev = loader.load(dev_path)
    print(dev.datasets['train'])
    # label_vocab = get_label_vocab('number)
    # print(label_vocab)
    # print(list(label_vocab))
    # pass
    # ds, vocabs, embeddings = load_aicup_ner(
    #     aicup_ner_path,
    #     yangjie_rich_pretrain_unigram_path,
    #     yangjie_rich_pretrain_bigram_path,
    #     index_token=False,
    #     char_min_freq=1,
    #     bigram_min_freq=1,
    #     only_train_min_freq=True,
    #     char_word_dropout=0.01,
    #     cv=True,
    #     model_type='many',
    #     fold=0,
    # )
    # print(vocabs['label'])
    
    # ds, vocabs, embeddings = load_aicup_ner(
    #     aicup_ner_path,
    #     yangjie_rich_pretrain_unigram_path,
    #     yangjie_rich_pretrain_bigram_path,
    #     index_token=True,
    #     char_min_freq=1,
    #     bigram_min_freq=1,
    #     only_train_min_freq=True,
    #     char_word_dropout=0.01,
    #     cv=True,
    #     fold=0,
    #     data_type='number'
    # )
    # print(ds)
