from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.utils import shuffle
from typing import Dict, List
from nlpcda import Ner
import numpy as np
import torch
import sys
import opencc


fpath = '/home/dy/flat-chinese-ner/data/train_2.txt'
USE_ALL_DATA_FOR_TRAIN = False
max_len=128
tagging_method = 'BI'
role_map = {
    '_' : 0, '*' : 0, '^' : 1,
    '&' : 1, '~' : 0, '@': 0, '(': 0,
    '=' :0, '%' :0
}
all_type = {
        'med_exam', 'money', 'contact', 'family',
        'clinical_event', 'location', 'ID', 'education',
        'others', 'name', 'time', 'profession', 'organization'
}
few_type = {
        'others', 'organization', 
        'clinical_event', 
}


def loadInputFile(path, filter_type=[]):  
    # The default filtering label list
    if filter_type == []:
        filter_type = list(few_type) 

    trainingset = list()
    position = list()
    mentions = dict()
    with open(path, 'r', encoding='utf8') as f:
        file_text=f.read().encode('utf-8').decode('utf-8-sig')
        
        converter = opencc.OpenCC('t2s.json')
        file_text = converter.convert(file_text)

    datas=file_text.split('\n\n--------------------\n\n')[:-1]
    for data in datas:
        data=data.split('\n')
        content=data[0]
        trainingset.append(content)
        annotations=data[1:]
        for annot in annotations[1:]:
            annot=annot.split('\t')
            if annot[4] in filter_type:
                continue
            position.extend(annot)
            mentions[annot[3]]=annot[4]
    
    return trainingset, position, mentions


def romove_redundant_str(article_doc, dev_mode=False):
    str_len = {
        '_' : 3, '*' : 4,
         '&' : 3, '^' : 3, '~': 4,
        '@' : 1, '(':6, '=' :4, '%':5
    }
    ori_len = len(article_doc)
    offset_map = np.array([0], dtype=np.int)

    article_doc = article_doc.replace('医师：', '_')
    article_doc = article_doc.replace('医师A：', '=')
    article_doc = article_doc.replace('医师B：', '=')
    article_doc = article_doc.replace('民众：', '^')
    article_doc = article_doc.replace('家属：', '&')
    article_doc = article_doc.replace('个管师：', '*')
    article_doc = article_doc.replace('护理师：', '~')
    article_doc = article_doc.replace('护理师A：', '%')
    article_doc = article_doc.replace('……', '…、')
    article_doc = article_doc.replace('…', '@')
    # article_doc = article_doc.replace('．．．．．．', '(')
  
    for word in article_doc:
        if word in str_len.keys():
            if dev_mode:
                offset_map = np.delete(offset_map, -1)
                # B.C. for offset_map at init
                last_offset =  offset_map[-1]  if offset_map.size > 0 else 0
                offset_map = np.append(offset_map, str_len[word] + last_offset )
            else:
                # need -1 since the return value include special tokens at training data
                offset_map = np.append(offset_map, str_len[word] + offset_map[-1] - 1 )
        else :
            offset_map = np.append(offset_map, offset_map[-1])

    # delete init value
    if dev_mode:
        article_doc = article_doc.replace('_', '')
        article_doc = article_doc.replace('*', '')
        article_doc = article_doc.replace('^', '')
        article_doc = article_doc.replace('&', '')
        article_doc = article_doc.replace('~', '')
        article_doc = article_doc.replace('@', '')
        article_doc = article_doc.replace('(', '')
    offset_map = np.delete(offset_map, 0)
    return article_doc, offset_map

token_continual_number = False


def preprocess_input(trainingset, position):
    # position is array like [article_id, start_pos, end_pos, type]
    label_position = 0
    input_id_types = []
    clean_docs = []
    label_doc = []
    for doc_idx, doc in enumerate(trainingset):
        label_queue = []
        doc, offset_map = romove_redundant_str(doc)
        input_id_types.append(generate_type_id(doc, offset_map))
        clean_sentences = ''
        label_per_doc = []
        
        for idx, word in enumerate(doc):

            if word in role_map.keys():
                # if label_queue:
                #     label_queue.pop(0)
                continue
            idx += offset_map[idx]
            if not label_position >= len(position) and idx == int(position[label_position + 1]) and doc_idx==int(position[label_position]):
                times = int(position[label_position + 2]) - int(position[label_position + 1])
                label = position[label_position + 4]
                label_queue.extend([label]*times)
                
                if tagging_method == 'BI':
                    label_queue = list(map(
                        lambda ele: f'B-{ele[1]}' if ele[0]==0 \
                        else f'I-{ele[1]}', enumerate(label_queue)))
                else:
                    if times ==1:
                        label_queue = ['S-'+label]
                    elif times ==2:
                        label_queue = list(map(
                            lambda ele: f'B-{ele[1]}' if ele[0]==0 \
                            else f'E-{ele[1]}', enumerate(label_queue)
                        ))
                    else:
                        label_queue_tmp = []
                        for i in range(len(label_queue)):
                            if i == 0:
                                label_queue_tmp.append(f'B-{label}')
                            elif i == len(label_queue)-1:
                                label_queue_tmp.append(f'E-{label}')
                            else:
                                label_queue_tmp.append(f'I-{label}')
                        label_queue = label_queue_tmp.copy()
                label_position += 5
            tag = 'O' if not label_queue else label_queue.pop(0)
            # if word == '…' or word == '．．．':
            #     continue 
            
            clean_sentences += word
            label_per_doc.append(tag)

        clean_docs.append(clean_sentences)
        label_doc.append(label_per_doc)

    return clean_docs, label_doc, input_id_types


def split_to_sentence(data:List[str], input_id_types, max_len, tags=None):
    # 2 is num of special token added by tokenizer
    max_len = max_len - 2 
    break_word = ['。', '，', '!']
    small_doc = []
    sentence = ''
    tmp = ''
    # split article into small piece of sentence
    for doc in data:
        for word in doc:
            ## append tag and word into tmp 
            tmp += word

            ## condiction checking for max_len
            if word in break_word:
                if len(sentence) + len(tmp) <= max_len:
                    sentence += tmp
                    tmp = ''
                else:
                    if len(sentence) > len(tmp):
                        small_doc.append(sentence)
                        sentence = ''
                    else:
                        small_doc.append(tmp)
                        tmp = ''
        else:
            if sentence:
                small_doc.append(sentence)
                sentence = ''
            if tmp:
                sentence += tmp
                tmp = ''
    # cut string to word array
    small_doc = cut_words(small_doc, tags)
    
    # cut input_id_types
    type_tensor = []
    if input_id_types: 
        type_flatten = [tag for list_tag in input_id_types for tag in list_tag ] 
        pos = 0
        for doc in small_doc:
            end_pos = pos + len(doc)
            type_tensor.append(type_flatten[pos:end_pos])
            pos = end_pos 
    
    doc_tags = []
    ## align tags when it is given (for training time)
    if tags != [] and tags != None:
        tags_flatten = [tag for list_tag in tags for tag in list_tag ]
        pos = 0
        for doc in small_doc:
            end_pos = pos + len(doc)
            doc_tags.append(tags_flatten[pos:end_pos])
            pos = end_pos
        return small_doc, doc_tags, type_tensor
    else:
    # prediction, no tags
        return small_doc, type_tensor


def cut_words(texts:List[str], tags=None) -> List[List[str]]:
    word_array = []
    tmp = ''
    for sid, sentences in enumerate(texts):
        cut_sentences = []
        for idx, word in enumerate(sentences):
            # if the last element of sentence and self is number
            # concat to that
            if token_continual_number and sentences[idx-1].isdigit() and word.isdigit():
                cut_sentences[-1] += word

            else:
                cut_sentences.append(word)

        word_array.append(cut_sentences)
    return word_array
    

def val_split(texts, tags, input_id_types, cv=False):
    from sklearn.model_selection import train_test_split
    

    SPLIT = 0.5
    #  this block is use for upload dev data to ai-cup platform 
    if USE_ALL_DATA_FOR_TRAIN:
        texts, tags = zip(*shuffle(list(zip(texts, tags))))
        return texts, tags, input_id_types, None, None, None

    text_ds = np.array(list(zip(texts, input_id_types)), dtype=object)
    # print(tags)
    if cv: SPLIT = .2
    train_texts, val_texts, train_tags, val_tags = train_test_split(text_ds, tags, test_size=SPLIT)

    if cv: return train_texts, val_texts, train_tags, val_tags

    val_texts, test_texts, val_tags, test_tags = train_test_split(val_texts, val_tags, test_size=.5)
    return train_texts, train_tags, val_texts, val_tags, test_texts, test_tags


def encode_data(
    texts,
    token_type_ids,
    text_tags=None,
    **kargs,
    ):
    from transformers import AutoTokenizer

    PAD_TOKEN = 'O'
    max_len = kargs.get('max_len')

    tokenizer = AutoTokenizer.from_pretrained(
        kargs.get('pretrained'),
        padding_side='right',
    )
    encodings = tokenizer(
        texts,
        is_split_into_words=True,
        padding=True, 
        max_length=max_len,
        return_token_type_ids=False,
        add_special_tokens=True,
    )  
    if kargs.get('use_type_id'):
        for i, input_ids in enumerate(encodings['input_ids']):
            len_input = len(input_ids)
            # CLS typeid
            token_type_ids[i].insert(0, 0)
            len_typeid = len(token_type_ids[i])
            # PAD typeid
            if len_typeid < max_len -1:
                token_type_ids[i].extend([0]*(max_len-len_typeid-1))
            # SEP typeid
            token_type_ids[i].append(0)
            assert(len(token_type_ids[i]) == max_len)

        encodings['token_type_ids'] = token_type_ids

    if text_tags:
        labels = [[tag2id[tag] for tag in doc] for doc in text_tags]
        # padding the label
        for i, batch_label in enumerate(labels):
            # CLS 
            batch_label = [tag2id[PAD_TOKEN]] + batch_label 
            # SEP 
            batch_label.append(tag2id[PAD_TOKEN]) 
            # PAD 
            batch_label.extend([tag2id[PAD_TOKEN]] *(max_len - len(batch_label)))
            labels[i] = batch_label
            assert(max_len == len(batch_label) == len(encodings['input_ids'][i]))
        return encodings, labels
    else:
        return encodings


class AICupDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        if self.labels:
            return len(self.labels)
        else:
            return len(self.encodings['input_ids'])


def get_dataset(**kargs):
    trainingset, position, _ = loadInputFile(fpath)

    texts, tags, input_id_types = preprocess_input(trainingset, position)
    texts, tags, input_id_types = split_to_sentence(texts, input_id_types, kargs.get('max_len'), tags=tags,)
    train_texts_ds, train_tags, val_texts_ds, val_tags, test_texts_ds, test_tags = val_split(texts, tags, input_id_types)

    train_texts, train_input_id_types = zip(*train_texts_ds)
    val_texts, val_input_id_types = zip(*val_texts_ds)
    test_texts, test_input_id_types = zip(*test_texts_ds)
    
    train_encodings, train_labels = encode_data(train_texts, train_input_id_types, train_tags, **kargs)
    train_dataset = AICupDataset(train_encodings, train_labels)
    # this block is used to upload data to ai cup platform
    if USE_ALL_DATA_FOR_TRAIN:
        return train_dataset, None, None

    val_encodings, val_labels = encode_data(val_texts, val_input_id_types, val_tags, **kargs)
    val_dataset = AICupDataset(val_encodings, val_labels)

    test_encodings, test_labels = encode_data(test_texts, test_input_id_types, test_tags, **kargs)
    test_dataset = AICupDataset(test_encodings, test_labels)

    return train_dataset, val_dataset, test_dataset


def generate_type_id(doc, offset_map):
    type_id = []
    for idx, word in enumerate(doc):
        if word in role_map.keys():
            offset = offset_map[idx]
            role = word
        else:
            if offset_map[idx] == offset:
                type_id.append(role_map[role])
    return type_id


def get_label(path='/home/dy/flat-chinese-ner/data/train_2.txt'):
    labels = list()
    with open(path, 'r', encoding='utf8') as f:
        file_text=f.read().encode('utf-8').decode('utf-8-sig')
    datas=file_text.split('\n\n--------------------\n\n')[:-1]
    for data in datas:
        data=data.split('\n')
        annotations=data[1:]
        for annot in annotations[1:]:
            annot=annot.split('\t')
            labels.append(annot[4])

    unique_labels = list(set(labels))
    tagging_labels = []

    if tagging_method == 'BI':
        for label in unique_labels:
            for prefix in tagging_method:
                tagging_labels.append(f'{prefix}-{label}')
    tagging_labels.append('O')
    return tagging_labels


def fix_BIOES_tag(fix_tag):
    for i in range(len(fix_tag)):
        for j in range(len(fix_tag[i])):
            if fix_tag[i][j][0] == 'B':
                # single
                if j==len(fix_tag[i])-1 or fix_tag[i][j+1][0] == 'O':
                    fix_tag[i][j] = 'S' + fix_tag[i][j][1:]
            elif fix_tag[i][j][0] == 'I':
                if j==len(fix_tag[i])-1 or fix_tag[i][j+1][0] == 'O':
                    fix_tag[i][j] = 'E' + fix_tag[i][j][1:]
    return fix_tag


def write_ds(outfile, texts, tags, split_sen=True):
    out = ''
    for a_id, doc in enumerate(texts):
        for idx, word in enumerate(doc):
            if word == ' ':
                continue
            # print(word, tags[a_id][idx])
            out += f'{word}\t{tags[a_id][idx]}\n' 
        if (len(doc) < 20):
            print(len(doc))
        if split_sen:
            out+='\n'
    
    with open(outfile, 'w+' ,encoding='utf-8') as f:
        f.write(out)


tag2id = {tag: id for id, tag in enumerate(get_label())}
id2tag = {id: tag for tag, id in tag2id.items()}


def filter_Otexts(texts, tags, aug_type):
    '''
    Filter the sentences that did't 
    include any given entity (aug_type)  
    
    Set aug_type to all_label to filter the sentences with all O tags
    '''
    aug_type = [    
        [f'B-{tag}', f'I-{tag}', f'S-{tag}'] for tag in aug_type 
    ]
    aug_type = [atype for sublist in aug_type for atype in sublist]
    
    filtered_texts = []
    filtered_tags = []
    for i, sentence_tag in enumerate(tags):
        have_entity=[]
        sentence_tag = np.array(sentence_tag)
        for entity_type in aug_type:
            have_entity.append(np.any(sentence_tag == entity_type))

        if not np.any(have_entity):
            pass
        else:
            filtered_texts.append(texts[i])
            filtered_tags.append(sentence_tag)

    return filtered_texts, filtered_tags


def augment(prefix, fold ,aug_type=[], augument_size=3):
    ner = Ner(
        ner_dir_name=prefix+'filtered',
        ignore_tag_list=['O'],
        data_augument_tag_list=aug_type,
        augument_size=augument_size, 
        seed=0
    )
    aug_texts, aug_tags = ner.augment(file_name=f'{prefix}filtered/raw')
    write_ds(f'./data/visualize/{str(aug_type)}{fold}', aug_texts, aug_tags)
    return aug_texts, aug_tags


HANDLE = 'default'
model_type = {
    'minority' : [
        'ID', 'education', 'family', 'profession',
    ],
    'number' : [
        'money', 'time', 'med_exam', 'ID', 
    ],
    'string' : [
        'money', 'time', 'contact', 'family',
        'location', 'education',
        'name', 'profession', 
    ],
    'default' : [],
    'time' : ['time'],
}
aug_size = 3
model_teamwork = False
remove_sentence_with_allO = False
use_pseudo = True

if __name__ == "__main__":
    '''
    1. Data =>> Training, Validation (Sentence)
    
        Select type of tags to Augmentation

    2. Filtered data =>> Filter the sentence without tags that need augmentation (Sentence)
    
    3. Augment data  =>> Augment the Filtered data (Raw)

    4. Concat Augment and Training data. (Sentence)

        Augment data need to split into sentence
    ''' 
    # 
    if HANDLE =='default':
        aug_type = ['family', 'location', 'education', 'profession', 'contact']
    else:
        aug_type = model_type[HANDLE]
    
    if model_teamwork:
        # disjoint
        filter_type = all_type - set(aug_type)
        trainingset, position, labels = loadInputFile(fpath, filter_type=filter_type)
    else:
        trainingset, position, labels = loadInputFile(fpath, filter_type=[])
    
    texts, tags, input_id_types = preprocess_input(trainingset, position)
    texts, tags, _ = split_to_sentence(texts, input_id_types, max_len, tags)

    # if not tagging_method == 'BI':
    #     tags = fix_BIOES_tag(tags)


    if use_pseudo:
        pseudo_set, pseudo_pos, _ = loadInputFile('./data/pseudo_data.txt', filter_type=[])
        pseudo_text, pseudo_tag, _ = preprocess_input(pseudo_set, pseudo_pos)
        pseudo_text, pseudo_tag = filter_Otexts(pseudo_text, pseudo_tag, list(all_type))
        pseudo_text, pseudo_tag, _ = split_to_sentence(pseudo_text, None, max_len, pseudo_tag)
        texts += pseudo_text
        tags += pseudo_tag
        # trainingset += pseudo_set
        # position += pseudo_pos
    texts = np.array(texts, dtype=object)
    tags = np.array(tags, dtype=object)

    print('Using label:', set(list(labels.values())))
    print('Origin sentence...', len(texts))
    # Kfold split
    kf = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    
    for idx, (train, test) in enumerate(kf.split(texts)):
        orgin_train, orgin_tags = texts[train].tolist(), tags[train].tolist()
        dev_text, dev_tags = texts[test], tags[test]
        if remove_sentence_with_allO:
            orgin_train, orgin_tags = filter_Otexts(orgin_train, orgin_tags, list(all_type))
            print('Sentence with valid tags', len(texts))


        print(f'--------Fold {idx}----------')
        filtered_texts, filtered_tags = filter_Otexts(orgin_train, orgin_tags, aug_type)

        prefix = f'./data/fold{idx}/'
        write_ds(f'{prefix}filtered/raw', filtered_texts, filtered_tags, split_sen=False)
        
        if aug_size==0:
            # Augmentation Disabled 
            write_ds(f'{prefix}/train/{HANDLE}', orgin_train, orgin_tags)
            write_ds(f'{prefix}dev/{HANDLE}', dev_text, dev_tags)
            continue
        
        aug_texts, aug_tags = augment(prefix, idx, aug_type=aug_type, augument_size=aug_size)
        write_ds(f'{prefix}dev/{HANDLE}', dev_text, dev_tags)
        

        aug_sen, sen_tags, _ = split_to_sentence(aug_texts, None, max_len, aug_tags)
        aug_sen, sen_tags = filter_Otexts(aug_sen, sen_tags, list(all_type))
        
        if not tagging_method =='BI':
            sen_tags = fix_BIOES_tag(sen_tags)

        print('Sentence to augmentation', len(filtered_texts))
        print('Augmented sentence', len(aug_sen))
        # concat augmented sample
        # print('Origin sentence...', len(orgin_train))

        orgin_train += aug_sen
        orgin_tags += sen_tags
        orgin_train, orgin_tags = zip(*shuffle(list(zip(orgin_train, orgin_tags))))
        write_ds(f'{prefix}/train/{HANDLE}', orgin_train, orgin_tags)
        print('After augmentation', len(orgin_train))

        print('train:',len(orgin_train),
            'val:',len(dev_text),)  
