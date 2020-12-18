from fastNLP_module import StaticEmbedding
from fastNLP.io.loader import ConllLoader
from fastNLP import cache_results
from fastNLP import Vocabulary
from fastNLP import DataSet
from utils import get_bigrams
import os
import sys
# sys.path.append('/home/dy/Flat-Lattice-Transformer/src')
from src.dataset import romove_redundant_str, split_to_sentence, cut_words, get_fastnlp_ds
from src.predict import load_dev
from paths import *


@cache_results(_cache_fp='cache/aicupNER_uni+bi', _refresh=True)
def load_aicup_ner(
    path,
    unigram_embedding_path=None,
    bigram_embedding_path=None,
    char_word_dropout=0.01,
    only_train_min_freq=0,
    bigram_min_freq=1,
    model_type='many',
    index_token=True,
    char_min_freq=1,
    cv=False,
    fold=0,
    ):
    vocabs = {}
    embeddings = {}

    train_path = os.path.join(path, f'fold{fold}', 'train/train')
    dev_path = os.path.join(path, f'fold{fold}', 'dev/dev',)
    print('loading data from', train_path,'\nand', dev_path)
    loader = ConllLoader(['chars', 'target'])

    train = loader.load(train_path)
    dev = loader.load(dev_path)
 
    ds = {
        'train':train.datasets['train'],
        'dev':dev.datasets['train'],
    }
    ds['aicup_dev'], offset_map = get_aicup_devds()

    for ds_name in ds.keys():
        ds[ds_name].apply_field(get_bigrams, 'chars', 'bigrams')
        ds[ds_name].add_seq_len('chars', new_field_name='seq_len')

    for k,v in ds.items():
        print('{}:{}'.format(k,len(v)))

    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()

    label_vocab.from_dataset(ds['train'], field_name='target')

    print(set([ele[0].split('-')[1] if ele[0]!='O' and ele[0][0]!='<' else ele[0] for ele in list(label_vocab)]))
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

    if index_token:
        char_vocab.index_dataset(*list(ds.values()), field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(*list(ds.values()),field_name='bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(*list([ds['train'], ds['dev']]), field_name='target', new_field_name='target')

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
        raw_data[idx], offset_map = romove_redundant_str(raw_data[idx], dev_mode=True)
        offset_mapping.append(offset_map)

    split_docs, type_tensor = split_to_sentence(raw_data, None, 128)

    dev_ds = DataSet({'chars':split_docs})    
    return dev_ds, offset_mapping


if __name__ == "__main__":
    
    train_path = os.path.join(aicup_ner_path, 'augment', f'train0')
    loader = ConllLoader(['chars', 'target'])

    train = loader.load(train_path)
 
    ds = train.datasets['train']
    print(list(ds['target']))
    # load_aicup_ner(
    #     aicup_ner_path,
    #     yangjie_rich_pretrain_unigram_path,
    #     yangjie_rich_pretrain_bigram_path,
    #     index_token=True,
    #     char_min_freq=1,
    #     bigram_min_freq=1,
    #     only_train_min_freq=True,
    #     char_word_dropout=0.01,
    #     cv=True,
    #     model_type='many',
    #     fold=0,
    # )
