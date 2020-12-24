from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.utils import shuffle
from typing import Dict, List
from nlpcda import Ner
import numpy as np
import torch
import sys
import opencc
from fastNLP import Vocabulary, DataSet


max_len=128
tagging_method = 'BI'
# tagging_method = 'BIES'
#TODO
token_continual_number = False
USE_ALL_DATA_FOR_TRAIN = False
fpath = '/home/dy/flat-chinese-ner/data/train_2.txt'
aug_size = 3
model_teamwork = False
remove_sentence_with_allO = False
use_pseudo = False
HANDLE = 'default'
role_map = {
    '_' : 0, '*' : 1, '^' : 1,
}
all_type = {
    'med_exam', 'money', 'contact', 'family',
    'clinical_event', 'location', 'ID', 'education',
    'others', 'name', 'time', 'profession', 'organization'
}
few_type = {
    'others', 'organization', 'clinical_event', 
}
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
    'default' : [ 
        'money', 'med_exam', 'profession', 'education', 'ID',
        'contact', 'family'
    ],
    'time' : ['time'],
}
break_word = ['。', '，', '!']

def loadInputFile(path):  
    converter = opencc.OpenCC('t2s.json')
    with open(path, 'r', encoding='utf8') as f:
        file_text=f.read().encode('utf-8').decode('utf-8-sig')   
        file_text = converter.convert(file_text)

    datas=file_text.split('\n\n--------------------\n\n')[:-1]
    return datas


def parse_document(datas, filter_type=[]):
    # The default filtering label list
    if filter_type == []:
        filter_type = list(few_type) 
    trainingset = list()
    position = list()
    mentions = dict()
    for data in datas:
        data=data.split('\n')
        content=data[0]
        trainingset.append(content)
        annotations=data[1:]
        for annot in annotations[1:]:
            annot=annot.split('\t')
            if annot[4] in filter_type:
                continue
            # print(annot[0], annot[3])
            position.extend(annot)
            mentions[annot[3]]=annot[4]
    
    return trainingset, position, mentions


def romove_redundant_str(article_doc, dev_mode=False):
    redundant_str = [
        '医师：','民众：','家属：',
        '个管师：', '护理师：', '医师A：', '医师B：'
    ]
    len_str = { '_' : 3, '*': 4  , '^' :5}
    str_len = { 3 : '_', 4 : '*' , 5 : '^'}

    for role_str in redundant_str:
        # Change to Sentence form by add 。
        article_doc = article_doc.replace('……'+role_str, '…。'+str_len[len(role_str)])
        article_doc = article_doc.replace(role_str, str_len[len(role_str)])

    offset_map = np.array([0], dtype=np.int)
    for word in article_doc:
        if word in len_str.keys():
            if dev_mode:
                offset_map = np.delete(offset_map, -1)
                # B.C. for offset_map at init
                last_offset =  offset_map[-1]  if offset_map.size > 0 else 0
                offset_map = np.append(offset_map, len_str[word] + last_offset )
            else:
                # need -1 since the return value include special tokens at training data
                offset_map = np.append(offset_map, len_str[word] + offset_map[-1] - 1 )
        else :
            offset_map = np.append(offset_map, offset_map[-1])

    # delete init value
    if dev_mode:
        for token in len_str.keys():
            article_doc = article_doc.replace(token, '')

    offset_map = np.delete(offset_map, 0)
    return article_doc, offset_map


def preprocess_input(trainingset, position, add_prefix=True):
    # position is array like [article_id, start_pos, end_pos, type]
    label_position = 0
    input_id_types = []
    clean_docs = []
    label_doc = []
    for doc in trainingset:
        label_queue = []
        doc, offset_map = romove_redundant_str(doc)
        input_id_types.append(generate_type_id(doc, offset_map))
        clean_sentences = ''
        label_per_doc = []
        
        for idx, word in enumerate(doc):
            if word in role_map.keys():
                continue
            idx += offset_map[idx]
            if label_position < len(position):
                doc_idx = int(position[label_position])
            
            if not label_position >= len(position) and \
                    idx==int(position[label_position + 1]) and \
                    doc_idx==int(position[label_position]):
                # DEBUG
                # print(doc_idx , position[label_position + 3])
                times = int(position[label_position + 2]) - int(position[label_position + 1])
                label = position[label_position + 4]
                label_queue.extend([label]*times)
                # Add the BIO or BIOES tagging prefix to label
                if add_prefix:
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
            # print(doc_idx, word, tag)
            clean_sentences += word
            label_per_doc.append(tag)
            
        clean_docs.append(clean_sentences)
        label_doc.append(label_per_doc)

    return clean_docs, label_doc, input_id_types


def split_to_sentence(data:List[str], input_id_types, max_len, tags=None, cut=True):
    # 2 is num of special token added by tokenizer
    max_len = max_len - 2 
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
    if cut:
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
            # if token_continual_number and sentences[idx-1].isdigit() and word.isdigit():
            #     cut_sentences[-1] += word
            # else:
            cut_sentences.append(word)

        word_array.append(cut_sentences)
    return word_array
    

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


def get_label_vocab(data_type='default'):
    label = [
            'family', 'education', 'money',
            'med_exam', 'ID', 'contact', 
            'name', 'time', 'location', 'profession'
    ]
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
            if False:
                print(len(doc))
        if split_sen:
            out+='\n'
    
    with open(outfile, 'w+' ,encoding='utf-8') as f:
        f.write(out)


def filter_Otexts(texts, tags, aug_type):
    '''
    Filter the sentences that did't 
    include any given entity (aug_type)  
    
    Set aug_type to all_label to filter the sentences with all O tags
    '''
    aug_type = [    
        [f'B-{tag}', f'I-{tag}'] for tag in aug_type 
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


def find_break_word(sens, mode=0):
    if mode == 0 :
        sens = list(reversed(sens))
    for idx, word in enumerate(sens):
        if word in break_word:
            return idx+1


def cut_to_max_len(sens, max_len):
    # 0 for head, 1 for tail    
    mode = 0
    s_pos, e_pos = 0, len(sens) 
    while (e_pos-s_pos) >= max_len:
        if mode:
            s_pos += find_break_word(sens[s_pos:e_pos], mode)
        else: 
            e_pos -= find_break_word(sens[s_pos:e_pos], mode)
        mode = not mode
    return s_pos, e_pos
    

def sliding_window(sens, tags, max_len):
    joined_sens = []
    joined_tags = []
    
    # combine the two sentence
    for idx, sentence in enumerate(sens):
        if idx == len(sens)-2:
            break
        joined_sens.append(sentence + sens[idx+1])
        joined_tags.append(tags[idx]+ tags[idx+1])

    # cut sentence by token 
    for idx, sentence in enumerate(joined_sens):
        s_pos, e_pos = cut_to_max_len(sentence, max_len)
        # print(sentence)
        joined_sens[idx] = sentence[s_pos:e_pos+1]
        joined_tags[idx] = joined_tags[idx][s_pos:e_pos+1]
        # print('hey')
        # print(sentence[s_pos:e_pos+1])
        # break
    # for i in range(len(sens)):
    # print('Single sentence-------')
    # print(sens[0])
    # print('Joined sentence-------')
    # print(joined_sens[0])
    return joined_sens, joined_tags


def augment(prefix, fold ,aug_type=[], augument_size=3):
    ner = Ner(
        ner_dir_name=prefix+'filtered',
        ignore_tag_list=['O'],
        data_augument_tag_list=aug_type,
        augument_size=augument_size, 
        seed=0
    )
    aug_texts, aug_tags = ner.augment(file_name=f'{prefix}filtered/raw')
    write_ds(f'./data/visualize/raw{fold}', aug_texts, aug_tags)
    return aug_texts, aug_tags


'''
    1. Data =>> Training, Validation (Sentence)
    
        Select type of tags to Augmentation

    2. Filtered data =>> Filter the sentence without tags that need augmentation (Sentence)
    
    3. Augment data  =>> Augment the Filtered data (Raw)

    4. Concat Augment and Training data. (Sentence)

        Augment data need to split into sentence
'''
if __name__ == "__main__":
        
    aug_type = model_type[HANDLE]
    
    # if model_teamwork:
    #     # disjoint
    #     filter_type = all_type - set(aug_type)
    #     docs = loadInputFile(fpath)
    #     trainingset, position, labels = parse_document(docs, filter_type=filter_type)
    # else:
    #     docs = loadInputFile(fpath)
    #     trainingset, position, labels = parse_document(docs, filter_type=[])
    
    # if not tagging_method == 'BI':
    #     tags = fix_BIOES_tag(tags)
    

        # pseudo_set, pseudo_pos, _ = parse_document(docs, filter_type=[]) 

    docs = loadInputFile(fpath)
    if use_pseudo:
        pseudo_docs = loadInputFile('./data/pseudo_data.txt')

    docs = np.array(docs) 
    print('Origin docs...', docs.shape[0])

    # Kfold split
    kf = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    for idx, (train, test) in enumerate(kf.split(docs)):
        train_docs, test_docs = docs[train].tolist(), docs[test].tolist()
        
        train_set, train_pos, _ = parse_document(train_docs, filter_type=[])
        test_set, test_pos, _   = parse_document(test_docs, filter_type=[])
        
        orgin_train, orgin_tags, _ = preprocess_input(train_set, train_pos)
        dev_text, dev_tags, _ =      preprocess_input(test_set, test_pos)
        
        orgin_train, orgin_tags, _ = split_to_sentence(orgin_train, None, max_len, orgin_tags, cut=False)
        dev_text, dev_tags, _ =      split_to_sentence(dev_text, None, max_len, dev_tags)
        # Only 3 sentence
        
        if remove_sentence_with_allO:
            orgin_train, orgin_tags = filter_Otexts(orgin_train, orgin_tags, list(all_type))
            print('Sentence with valid tags', len(texts))

        
        print(f'--------Fold {idx}----------')
        prefix = f'./data/fold{idx}/'
        write_ds(f'{prefix}dev/{HANDLE}', dev_text, dev_tags)

        print('Origin sentence', len(orgin_train))

        # Augmentation Disabled 
        if aug_size==0:

            write_ds(f'{prefix}/train/{HANDLE}', orgin_train, orgin_tags)
            continue
        
        # Data Augmentation
        Aug_train, Aug_tag = sliding_window(orgin_train, orgin_tags, max_len)
        # sliding_train, sliding_tags = sliding_window(orgin_train, orgin_tags, max_len)
  
        filtered_texts, filtered_tags = filter_Otexts(Aug_train, Aug_tag, aug_type)
        write_ds(f'{prefix}filtered/raw', filtered_texts, filtered_tags, split_sen=False)
        
        aug_texts, aug_tags = augment(prefix, idx, aug_type=aug_type, augument_size=aug_size)
        aug_sen, sen_tags, _ = split_to_sentence(aug_texts, None, max_len, aug_tags)
        aug_sen, sen_tags = filter_Otexts(aug_sen, sen_tags, list(all_type))
       
        # Sliding window Augmentation
        # sliding_train, sliding_tags = sliding_window(orgin_train, orgin_tags, max_len)
        # sliding_train, sliding_tags = filter_Otexts(sliding_train, sliding_tags, aug_type)
        # orgin_train += sliding_train
        # orgin_tags += sliding_tags
        
        # Change tagging method
        if not tagging_method =='BI':
            sen_tags = fix_BIOES_tag(sen_tags)

        print('Sentence to augmentation', len(filtered_texts))
        print('Augmented sentence', len(aug_sen))
        
        # Concat augmented sample
        orgin_train += aug_sen
        orgin_tags += sen_tags
        orgin_train, orgin_tags = zip(*shuffle(list(zip(orgin_train, orgin_tags))))
        write_ds(f'{prefix}/train/{HANDLE}', orgin_train, orgin_tags)
        
        print('train:',len(orgin_train),
                'val:',len(dev_text),)  

    # DEBUG
    if True:
        write_ds('./debug_sliding.txt', aug_sen, sen_tags)
        write_ds('./debug.txt', orgin_train, orgin_tags)
        