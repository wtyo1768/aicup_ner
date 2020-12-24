from src.dataset import *
import torch


def get_label(path='/home/dy/Flat-Lattice-Transformer/data/train_2.txt'):
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


tag2id = {tag: id for id, tag in enumerate(get_label())}
id2tag = {id: tag for tag, id in tag2id.items()}
