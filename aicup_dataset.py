from fastNLP_module import StaticEmbedding
from fastNLP.io.loader import ConllLoader
from fastNLP import cache_results
from fastNLP import Vocabulary
from fastNLP import DataSet
from utils import get_bigrams
import os
import sys
sys.path.append('/home/dy/aicup/src')
from dataset import romove_redundant_str, split_to_sentence, cut_words, get_fastnlp_ds
from predict import load_dev
from paths import *


@cache_results(_cache_fp='cache/aicupNER_uni+bi', _refresh=True)
def load_aicup_ner(
    path,
    unigram_embedding_path=None,
    bigram_embedding_path=None,
    index_token=True,
    char_min_freq=1,
    bigram_min_freq=1,
    only_train_min_freq=0,
    char_word_dropout=0.01,
    cv=False,
    model_type='many',
    ):

    vocabs = {}
    embeddings = {}
    if cv:
        train_texts, train_tags, val_texts, val_tags = get_fastnlp_ds(mode=model_type, cv=cv)
    else:     
        train_texts, train_tags, val_texts, val_tags, test_texts, test_tags = get_fastnlp_ds(mode=model_type, cv=cv)
    
    train_ds = DataSet({'chars':train_texts, 'target':train_tags})
    dev_ds = DataSet({'chars':val_texts, 'target':val_tags})
    ds = {
        'train':train_ds,
        'dev':dev_ds,
    }
    if not cv:
        ds['test'] = DataSet({'chars':test_texts, 'target':test_tags})
    ds['aicup_dev'], offset_map = get_aicup_devds()

    # loader = ConllLoader(['chars', 'target'])
    # for ds_name in ['train', 'dev', 'test']:
        # bundle = loader.load(os.path.join(path, ds_name))
        # ds[ds_name] = bundle.datasets['train']


    for ds_name in ds.keys():
        ds[ds_name].apply_field(get_bigrams, 'chars', 'bigrams')
        ds[ds_name].add_seq_len('chars', new_field_name='seq_len')

    for k,v in ds.items():
        print('{}:{}'.format(k,len(v)))

    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()

    label_vocab.from_dataset(ds['train'], field_name='target')

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

    # convert words to index in the vocab
    if index_token:
        char_vocab.index_dataset(*list(ds.values()), field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(*list(ds.values()),field_name='bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(*list(ds.values()), field_name='target', new_field_name='target')


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

    return ds, vocabs, embeddings, offset_map


def get_aicup_devds():
    raw_data = load_dev()
    
    offset_mapping = []
    for idx in range(len(raw_data)):
        # print(idx)
        raw_data[idx], offset_map = romove_redundant_str(raw_data[idx], dev_mode=True)
        offset_mapping.append(offset_map)

    split_docs, type_tensor = split_to_sentence(raw_data, None, 128)

    # import opencc
    # converter = opencc.OpenCC('t2s.json')

    # for idx in range(len(split_docs)):
    #     split_docs[idx] = converter.convert(split_docs[idx])

    # split_docs = cut_words(split_docs)
    dev_ds = DataSet({'chars':split_docs})    
    return dev_ds, offset_mapping


if __name__ == "__main__":

    # ds = get_aicup_devds()
    # ds.add_seq_len('chars', new_field_name='seq_len')
    # print(ds)

    loader = ConllLoader(['chars', 'target'], dropna=False)
    for ds_name in ['train', 'dev', 'test']:
        bundle = loader.load(os.path.join(root_path,'data', ds_name))
        ds[ds_name] = bundle.datasets['train']
    print(ds[ds_name])
    # from paths import *   

    # ds, vocabs, embeddings = load_aicup_ner(
    #     aicup_ner_path,
    #     yangjie_rich_pretrain_unigram_path,
    #     yangjie_rich_pretrain_bigram_path,
    #     _refresh=True,
    #     index_token=False,
    #     _cache_fp='./cache',
    #     char_min_freq=1,
    #     bigram_min_freq=1,
    #     only_train_min_freq=1
    # )
    # print(ds['train'])