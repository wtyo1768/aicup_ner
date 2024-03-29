import fitlog
use_fitlog = True
if not use_fitlog:
    fitlog.debug()
fitlog.set_log_dir('logs')
load_dataset_seed = 100
fitlog.add_hyper(load_dataset_seed,'load_dataset_seed')
fitlog.set_rng_seed(load_dataset_seed)
import sys
sys.path.append('../')

from src.dataset  import romove_redundant_str, write_ds
from src.predict import load_dev, split_to_pred_per_article, write_result
from src.predict import convert_pred_and_write, count_article_length
from load_data import *
import argparse
from paths import *
from fastNLP.core import Trainer
from fastNLP.core import Callback
from fastNLP.core.metrics import MetricBase
from V1.models import Lattice_Transformer_SeqLabel, Transformer_SeqLabel
import torch
import collections
import torch.optim as optim
import torch.nn as nn
from fastNLP import LossInForward
from fastNLP.core.metrics import SpanFPreRecMetric,AccuracyMetric
from fastNLP.core.callback import WarmupCallback,GradientClipCallback,EarlyStopCallback,SaveModelCallback
from fastNLP import FitlogCallback
from fastNLP import LRScheduler
from fastNLP.core.predictor import Predictor
from torch.optim.lr_scheduler import LambdaLR
import fitlog
from fastNLP import logger
from utils import get_peking_time
from V1.add_lattice import equip_chinese_ner_with_lexicon

import traceback
import warnings
import sys
from utils import print_info
# from fastNLP.embeddings import BertEmbedding
from fastNLP_module import BertEmbedding
from V1.models import BERT_SeqLabel
from seqeval.metrics import classification_report


parser = argparse.ArgumentParser()
# performance inrelevant
parser.add_argument('--cv',type=bool,default=False)
parser.add_argument('--crf_lr',type=float,default=0.1)
parser.add_argument('--pred_dir',type=str,required=True)
parser.add_argument('--data_type',type=str,default='few')
parser.add_argument('--fold',type=int,default=0)
parser.add_argument('--use_pos_tag', type=int, default=0, required=True)
parser.add_argument('--do_pred', type=int, default=0)
parser.add_argument('--update_every',type=int,default=1)
parser.add_argument('--status',choices=['train','test', 'bagging'],default='train')
parser.add_argument('--use_bert',type=int,default=1)
parser.add_argument('--only_bert',type=int,default=0)
parser.add_argument('--fix_bert_epoch',type=int,default=20)
parser.add_argument('--fix_embed_epoch',type=int,default=10)
parser.add_argument('--after_bert',default='mlp',choices=['lstm','mlp'])
parser.add_argument('--msg',default='11266')
parser.add_argument('--train_clip',default=False,help='是不是要把train的char长度限制在200以内')
parser.add_argument('--device', default='0')
parser.add_argument('--debug', default=0,type=int)
parser.add_argument('--gpumm',default=False,help='查看显存')
parser.add_argument('--see_convergence',default=False)
parser.add_argument('--see_param',default=False)
parser.add_argument('--test_batch', default=-1)
parser.add_argument('--seed', default=1080956,type=int)
parser.add_argument('--test_train',default=False)
parser.add_argument('--number_normalized',type=int,default=0,
                    choices=[0,1,2,3],help='0不norm，1只norm char,2norm char和bigram，3norm char，bigram和lattice')
parser.add_argument('--lexicon_name',default='yj',choices=['lk','yj'])
parser.add_argument('--use_pytorch_dropout',type=int,default=0)

parser.add_argument('--char_min_freq',default=1,type=int)
parser.add_argument('--bigram_min_freq',default=1,type=int)
parser.add_argument('--lattice_min_freq',default=1,type=int)
parser.add_argument('--only_train_min_freq',default=True)
parser.add_argument('--only_lexicon_in_train',default=False)


parser.add_argument('--word_min_freq',default=1,type=int)

# hyper of training
parser.add_argument('--early_stop',default=20,type=int)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--batch', default=32, type=int)
parser.add_argument('--optim', default='sgd', help='sgd|adam')
parser.add_argument('--lr', default=6e-4, type=float)
parser.add_argument('--bert_lr_rate',default=0.05,type=float)
parser.add_argument('--embed_lr_rate',default=1.5,type=float)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--init',default='uniform',help='norm|uniform')
parser.add_argument('--self_supervised',default=False)
parser.add_argument('--weight_decay',default=0,type=float)
parser.add_argument('--norm_embed',default=True)
parser.add_argument('--norm_lattice_embed',default=True)

parser.add_argument('--warmup',default=0.1,type=float)

# hyper of model
parser.add_argument('--model',default='transformer',help='lstm|transformer')
parser.add_argument('--lattice',default=1,type=int)
parser.add_argument('--use_bigram', default=1,type=int)
parser.add_argument('--hidden', default=-1,type=int)
parser.add_argument('--ff', default=3,type=int)
parser.add_argument('--layer', default=1,type=int)
parser.add_argument('--head', default=8,type=int)
parser.add_argument('--head_dim',default=20,type=int)
parser.add_argument('--scaled',default=False)
parser.add_argument('--ff_activate',default='relu',help='leaky|relu')

parser.add_argument('--k_proj',default=False)
parser.add_argument('--q_proj',default=True)
parser.add_argument('--v_proj',default=True)
parser.add_argument('--r_proj',default=True)

parser.add_argument('--attn_ff',default=False)

parser.add_argument('--use_abs_pos',default=False)
parser.add_argument('--use_rel_pos',default=True)
#相对位置和绝对位置不是对立的，可以同时使用
parser.add_argument('--rel_pos_shared',default=True)
parser.add_argument('--add_pos', default=False)
parser.add_argument('--learn_pos', default=False)
parser.add_argument('--pos_norm',default=False)
parser.add_argument('--rel_pos_init',default=1)
parser.add_argument('--four_pos_shared',default=True,help='只针对相对位置编码，指4个位置编码是不是共享权重')
parser.add_argument('--four_pos_fusion',default='ff_two',choices=['ff','attn','gate','ff_two','ff_linear'],
                    help='ff就是输入带非线性隐层的全连接，'
                         'attn就是先计算出对每个位置编码的加权，然后求加权和'
                         'gate和attn类似，只不过就是计算的加权多了一个维度')

parser.add_argument('--four_pos_fusion_shared',default=True,help='是不是要共享4个位置融合之后形成的pos')

# parser.add_argument('--rel_pos_scale',default=2,help='在lattice且用相对位置编码时，由于中间过程消耗显存过大，'
#                                                  '所以可以使4个位置的初始embedding size缩小，'
#                                                  '最后融合时回到正常的hidden size即可')

parser.add_argument('--pre', default='')
parser.add_argument('--post', default='an')

over_all_dropout =  -1
parser.add_argument('--embed_dropout_before_pos',default=False)
parser.add_argument('--embed_dropout', default=0.5,type=float)
parser.add_argument('--gaz_dropout',default=0.5,type=float)
parser.add_argument('--output_dropout', default=0.3,type=float)
parser.add_argument('--pre_dropout', default=0.5,type=float)
parser.add_argument('--post_dropout', default=0.3,type=float)
parser.add_argument('--ff_dropout', default=0.15,type=float)
parser.add_argument('--ff_dropout_2', default=-1,type=float,help='FF第二层过完后的dropout，之前没管这个的时候是0')
parser.add_argument('--attn_dropout',default=0,type=float)
parser.add_argument('--embed_dropout_pos',default='0')
parser.add_argument('--abs_pos_fusion_func',default='nonlinear_add',
                    choices=['add','concat','nonlinear_concat','nonlinear_add','concat_nonlinear','add_nonlinear'])

parser.add_argument('--dataset', default='ontonotes', help='weibo|resume|ontonotes|msra|aicup')




args = parser.parse_args()

if args.ff_dropout_2 < 0:
    args.ff_dropout_2 = args.ff_dropout

if over_all_dropout>0:
    args.embed_dropout = over_all_dropout
    args.output_dropout = over_all_dropout
    args.pre_dropout = over_all_dropout
    args.post_dropout = over_all_dropout
    args.ff_dropout = over_all_dropout
    args.attn_dropout = over_all_dropout


if args.lattice and args.use_rel_pos:
    args.train_clip = True

fitlog.set_log_dir('logs')
now_time = get_peking_time()
logger.add_file('log/{}'.format(now_time),level='info')
if args.test_batch == -1:
    args.test_batch = args.batch//2
fitlog.add_hyper(now_time,'time')
if args.debug:
    pass


if args.device!='cpu':
    assert args.device.isdigit()
    device = torch.device('cuda:{}'.format(args.device))
else:
    device = torch.device('cpu')

refresh_data = False

for k,v in args.__dict__.items():
    print_info('{} : {}'.format(k,v))

raw_dataset_cache_name = os.path.join('cache',args.data_type
                                      +'k{}'.format(args.fold)
                                      +'_cv:{}'.format(args.cv)
                                      +'_trainClip:{}'.format(args.train_clip) 
                                      +'bgminfreq_{}'.format(args.bigram_min_freq)
                                      +'char_min_freq_{}'.format(args.char_min_freq)
                                      +'word_min_freq_{}'.format(args.word_min_freq)
                                      +'only_train_min_freq{}'.format(args.only_train_min_freq)
                                      +'number_norm{}'.format(args.number_normalized)
                                      +'load_dataset_seed{}'.format(load_dataset_seed))


if args.dataset == 'weibo':
    datasets,vocabs,embeddings = load_weibo_ner(weibo_ner_path,yangjie_rich_pretrain_unigram_path,yangjie_rich_pretrain_bigram_path,
                                                _refresh=refresh_data,index_token=False,
                                                _cache_fp=raw_dataset_cache_name,
                                                char_min_freq=args.char_min_freq,
                                                bigram_min_freq=args.bigram_min_freq,
                                                only_train_min_freq=args.only_train_min_freq,
                                            )
elif args.dataset == 'aicup':
    from aicup_dataset import load_aicup_ner
    datasets,vocabs,embeddings = load_aicup_ner(aicup_ner_path,yangjie_rich_pretrain_unigram_path,yangjie_rich_pretrain_bigram_path,
                                                _refresh=refresh_data,index_token=False,
                                                char_word_dropout=0.05,
                                                _cache_fp=raw_dataset_cache_name,
                                                char_min_freq=args.char_min_freq,
                                                bigram_min_freq=args.bigram_min_freq,
                                                only_train_min_freq=args.only_train_min_freq,
                                                cv=args.cv,
                                                data_type=args.data_type,
                                                fold=args.fold,
                                            )

if args.gaz_dropout < 0:
    args.gaz_dropout = args.embed_dropout

args.hidden = args.head_dim * args.head
args.ff = args.hidden * args.ff


if args.lexicon_name == 'lk':
    yangjie_rich_pretrain_word_path = lk_word_path_2

print('用的词表的路径:{}'.format(yangjie_rich_pretrain_word_path))

w_list = load_yangjie_rich_pretrain_word_list(yangjie_rich_pretrain_word_path,
                                              _refresh=refresh_data,
                                              _cache_fp='cache/{}'.format(args.lexicon_name))

cache_name = os.path.join('cache', args.data_type
                                    +'k{}'.format(args.fold)
                                    +'_cv:{}'.format(args.cv)
                                    +'_lattice'+'_only_train:{}'
                                    +'_trainClip:{}'+'_norm_num:{}'
                                    +'char_min_freq{}'+'bigram_min_freq{}'
                                    +'word_min_freq{}'+'only_train_min_freq{}'
                                    +'number_norm{}'+'lexicon_{}'
                                    +'load_dataset_seed_{}').format(args.only_lexicon_in_train,
                          args.train_clip,args.number_normalized,args.char_min_freq,
                                  args.bigram_min_freq,args.word_min_freq,args.only_train_min_freq,
                                  args.number_normalized,args.lexicon_name,load_dataset_seed)
datasets,vocabs,embeddings = equip_chinese_ner_with_lexicon(datasets,vocabs,embeddings,
                                                            w_list,yangjie_rich_pretrain_word_path,
                                                         _refresh=refresh_data,_cache_fp=cache_name,
                                                         only_lexicon_in_train=args.only_lexicon_in_train,
                                                            word_char_mix_embedding_path=yangjie_rich_pretrain_char_and_word_path,
                                                            number_normalized=args.number_normalized,
                                                            lattice_min_freq=args.lattice_min_freq,
                                                            only_train_min_freq=args.only_train_min_freq,
                                                            cv=args.cv)
if args.status=='train':    
    print('train:{}'.format(len(datasets['train'])))
avg_seq_len = 0
avg_lex_num = 0
avg_seq_lex = 0
train_seq_lex = []
dev_seq_lex = []
test_seq_lex = []
train_seq = []
dev_seq = []
test_seq = []
for k,v in datasets.items():
    max_seq_len = 0
    max_lex_num = 0
    max_seq_lex = 0
    max_seq_len_i = -1
    for i in range(len(v)):
        if max_seq_len < v[i]['seq_len']:
            max_seq_len = v[i]['seq_len']
            max_seq_len_i = i
        max_lex_num = max(max_lex_num,v[i]['lex_num'])
        max_seq_lex = max(max_seq_lex,v[i]['lex_num']+v[i]['seq_len'])

        avg_seq_len+=v[i]['seq_len']
        avg_lex_num+=v[i]['lex_num']
        avg_seq_lex+=(v[i]['seq_len']+v[i]['lex_num'])
        if k == 'train':
            train_seq_lex.append(v[i]['lex_num']+v[i]['seq_len'])
            train_seq.append(v[i]['seq_len'])
            if v[i]['seq_len'] >200:
                print('train里这个句子char长度已经超了200了')
                print(''.join(list(map(lambda x:vocabs['char'].to_word(x),v[i]['chars']))))
            else:
                if v[i]['seq_len']+v[i]['lex_num']>400:
                    print('train里这个句子char长度没超200，但是总长度超了400')
                    print(''.join(list(map(lambda x: vocabs['char'].to_word(x), v[i]['chars']))))
        if k == 'dev':
            dev_seq_lex.append(v[i]['lex_num']+v[i]['seq_len'])
            dev_seq.append(v[i]['seq_len'])
        if k == 'test':
            test_seq_lex.append(v[i]['lex_num']+v[i]['seq_len'])
            test_seq.append(v[i]['seq_len'])


    print('{} 最长的句子是:{}'.format(k,list(map(lambda x:vocabs['char'].to_word(x),v[max_seq_len_i]['chars']))))
    print('{} max_seq_len:{}'.format(k,max_seq_len))
    print('{} max_lex_num:{}'.format(k, max_lex_num))
    print('{} max_seq_lex:{}'.format(k, max_seq_lex))


import copy
max_seq_len = max(* map(lambda x:max(x['seq_len']),datasets.values()))


if args.status=='train':
    show_index = 4
    print('raw_chars:{}'.format(list(datasets['train'][show_index]['raw_chars'])))
    print('lexicons:{}'.format(list(datasets['train'][show_index]['lexicons'])))
    print('lattice:{}'.format(list(datasets['train'][show_index]['lattice'])))
    print('raw_lattice:{}'.format(list(map(lambda x:vocabs['lattice'].to_word(x),
                                      list(datasets['train'][show_index]['lattice'])))))
    print('lex_s:{}'.format(list(datasets['train'][show_index]['lex_s'])))
    print('lex_e:{}'.format(list(datasets['train'][show_index]['lex_e'])))
    print('pos_s:{}'.format(list(datasets['train'][show_index]['pos_s'])))
    print('pos_e:{}'.format(list(datasets['train'][show_index]['pos_e'])))


for k, v in datasets.items():
    if args.lattice:
        v.set_input('lex_num','pos_s','pos_e')
        v.set_pad_val('lattice',vocabs['lattice'].padding_idx)
        if not k=='aicup_dev':
            v.set_target('target','seq_len')
            v.set_input('lattice','bigrams','seq_len','target')
        else:
            v.set_input('lattice','bigrams','seq_len',)
    else:
        if not k=='aicup_dev':
            v.set_target('target', 'seq_len')
            v.set_input('chars','bigrams','seq_len','target')
        else:
            v.set_input('chars','bigrams','seq_len')
    v.set_input('pos_tag')


from utils import norm_static_embedding
# print(embeddings['char'].embedding.weight[:10])
if args.norm_embed>0:
    print('embedding:{}'.format(embeddings['char'].embedding.weight.size()))
    print('norm embedding')
    for k,v in embeddings.items():
        norm_static_embedding(v,args.norm_embed)

if args.norm_lattice_embed>0:
    print('embedding:{}'.format(embeddings['lattice'].embedding.weight.size()))
    print('norm lattice embedding')
    for k,v in embeddings.items():
        norm_static_embedding(v,args.norm_embed)


mode = {}
mode['debug'] = args.debug
mode['gpumm'] = args.gpumm
if args.debug or args.gpumm:
    fitlog.debug()
dropout = collections.defaultdict(int)
dropout['embed'] = args.embed_dropout
dropout['gaz'] = args.gaz_dropout
dropout['output'] = args.output_dropout
dropout['pre'] = args.pre_dropout
dropout['post'] = args.post_dropout
dropout['ff'] = args.ff_dropout
dropout['ff_2'] = args.ff_dropout_2
dropout['attn'] = args.attn_dropout

torch.backends.cudnn.benchmark = False
fitlog.set_rng_seed(args.seed)
torch.backends.cudnn.benchmark = False

fitlog.add_hyper(args)

if args.model == 'transformer':
    if args.lattice:
        if args.use_bert:
            bert_embedding = BertEmbedding(
                vocabs['lattice'],
                model_dir_or_name='cn-wwm',
                requires_grad=False,
                word_dropout=0.02,
                layers='-1',
            )
        else:
            bert_embedding = None
        if args.only_bert:
            model = BERT_SeqLabel(
                bert_embedding,
                len(vocabs['label']),
                vocabs,
                args.after_bert,
                use_pos_tag=args.use_pos_tag
            )
        else:
            model = Lattice_Transformer_SeqLabel(embeddings['lattice'], embeddings['bigram'], args.hidden, len(vocabs['label']),
                                         args.head, args.layer, args.use_abs_pos,args.use_rel_pos,
                                         args.learn_pos, args.add_pos,
                                         args.pre, args.post, args.ff, args.scaled,dropout,args.use_bigram,
                                         mode,device,vocabs,
                                         max_seq_len=max_seq_len,
                                         rel_pos_shared=args.rel_pos_shared,
                                         k_proj=args.k_proj,
                                         q_proj=args.q_proj,
                                         v_proj=args.v_proj,
                                         r_proj=args.r_proj,
                                         self_supervised=args.self_supervised,
                                         attn_ff=args.attn_ff,
                                         pos_norm=args.pos_norm,
                                         ff_activate=args.ff_activate,
                                         abs_pos_fusion_func=args.abs_pos_fusion_func,
                                         embed_dropout_pos=args.embed_dropout_pos,
                                         four_pos_shared=args.four_pos_shared,
                                         four_pos_fusion=args.four_pos_fusion,
                                         four_pos_fusion_shared=args.four_pos_fusion_shared,
                                         bert_embedding=bert_embedding,
                                         use_pos_tag=args.use_pos_tag,
                                         after_bert=args.after_bert
                                         )
    else:
        model = Transformer_SeqLabel(embeddings['lattice'], embeddings['bigram'], args.hidden, len(vocabs['label']),
                                     args.head, args.layer, args.use_abs_pos,args.use_rel_pos,
                                     args.learn_pos, args.add_pos,
                                     args.pre, args.post, args.ff, args.scaled,dropout,args.use_bigram,
                                     mode,device,vocabs,
                                     max_seq_len=max_seq_len,
                                     rel_pos_shared=args.rel_pos_shared,
                                     k_proj=args.k_proj,
                                     q_proj=args.q_proj,
                                     v_proj=args.v_proj,
                                     r_proj=args.r_proj,
                                     self_supervised=args.self_supervised,
                                     attn_ff=args.attn_ff,
                                     pos_norm=args.pos_norm,
                                     ff_activate=args.ff_activate,
                                     abs_pos_fusion_func=args.abs_pos_fusion_func,
                                     embed_dropout_pos=args.embed_dropout_pos
                                     )


elif args.model =='lstm':
    model = LSTM_SeqLabel(embeddings['char'],embeddings['bigram'],embeddings['bigram'],args.hidden,
                               len(vocabs['label']),
                          bidirectional=True,
                          device=device,
                          embed_dropout=args.embed_dropout,output_dropout=args.output_dropout,use_bigram=True,
                          debug=args.debug)

if args.status=='train':
    for n,p in model.named_parameters():
        print('{}:{}'.format(n,p.size()))

    with torch.no_grad():
        print_info('{}init pram{}'.format('*'*15,'*'*15))
        for n,p in model.named_parameters():
            if 'bert' not in n and 'embedding' not in n and 'pos' not in n and 'pe' not in n \
                    and 'bias' not in n and 'crf' not in n and p.dim()>1:
                try:
                    if args.init == 'uniform':
                        nn.init.xavier_uniform_(p)
                        print_info('xavier uniform init:{}'.format(n))
                    elif args.init == 'norm':
                        print_info('xavier norm init:{}'.format(n))
                        nn.init.xavier_normal_(p)
                except:
                    print_info(n)
                    exit(1208)
        print_info('{}init pram{}'.format('*' * 15, '*' * 15))

loss = LossInForward()
# encoding_type = 'bioes'
encoding_type = 'bio'
f1_metric = SpanFPreRecMetric(
    vocabs['label'],
    pred='pred',
    target='target',
    seq_len='seq_len',
    encoding_type=encoding_type
)
acc_metric = AccuracyMetric(pred='pred',target='target',seq_len='seq_len',)
acc_metric.set_metric_name('label_acc')
metrics = [
    f1_metric,
    acc_metric,
]

if args.self_supervised:
    chars_acc_metric = AccuracyMetric(pred='chars_pred',target='chars_target',seq_len='seq_len')
    chars_acc_metric.set_metric_name('chars_acc')
    metrics.append(chars_acc_metric)

print('layernum:', args.layer)

if not args.only_bert:
    if not args.use_bert:
        bigram_embedding_param = list(model.bigram_embed.parameters())
        gaz_embedding_param = list(model.lattice_embed.parameters())
        embedding_param = bigram_embedding_param
        if args.lattice:
            gaz_embedding_param = list(model.lattice_embed.parameters())
            embedding_param = embedding_param+gaz_embedding_param
        embedding_param_ids = list(map(id,embedding_param))
        non_embedding_param = list(filter(lambda x:id(x) not in embedding_param_ids,model.parameters()))
        param_ = [{'params': non_embedding_param}, {'params': embedding_param, 'lr': args.lr * args.embed_lr_rate}]
    else:
        bert_embedding_param = list(model.bert_embedding.parameters())
        bert_embedding_param_ids = list(map(id,bert_embedding_param))
        bigram_embedding_param = list(model.bigram_embed.parameters())
        gaz_embedding_param = list(model.lattice_embed.parameters())
        embedding_param = bigram_embedding_param
        crf_param = list(model.crf.parameters())
        crf_param_ids = list(map(id,crf_param))

        if args.lattice:
            gaz_embedding_param = list(model.lattice_embed.parameters())
            embedding_param = embedding_param+gaz_embedding_param
        embedding_param_ids = list(map(id,embedding_param))
        non_embedding_param = list(filter(
            lambda x:id(x) not in embedding_param_ids and id(x) not in bert_embedding_param_ids and  id(x) not in crf_param_ids,
                                          model.parameters()))
        param_ = [
            {'params': non_embedding_param}, 
            {'params': embedding_param, 'lr': args.lr * args.embed_lr_rate},
            {'params':bert_embedding_param,'lr':args.bert_lr_rate*args.lr},
            {'params':crf_param, 'lr': args.crf_lr},
        ]
else:
    bert_embedding_param = list(model.bert_embedding.parameters())
    bert_embedding_param_ids = list(map(id,bert_embedding_param))
    crf_param = list(model.crf.parameters())
    crf_param_ids = list(map(id,crf_param))

    non_embedding_param = list(filter(
    lambda x: id(x) not in bert_embedding_param_ids and  id(x) not in crf_param_ids,
        model.parameters()))
    param_ = [
        {'params': non_embedding_param},
        {'params':bert_embedding_param,'lr':args.bert_lr_rate*args.lr},
        {'params':crf_param, 'lr': args.crf_lr},
    ]


if args.optim == 'adam':
    optimizer = optim.AdamW(
        param_,
        lr=args.lr,
        weight_decay=args.weight_decay)
elif args.optim == 'sgd':
    optimizer = optim.SGD(param_,lr=args.lr,momentum=args.momentum,
                          weight_decay=args.weight_decay)


class Unfreeze_Callback(Callback):
    def __init__(self,bert_embedding,fix_epoch_num):
        super().__init__()
        self.bert_embedding = bert_embedding
        self.fix_epoch_num = fix_epoch_num
        assert self.bert_embedding.requires_grad == False

    def on_epoch_begin(self):
        if self.epoch == self.fix_epoch_num+1:
            self.bert_embedding.requires_grad = True


def create_cb():
    lrschedule_callback = LRScheduler(lr_scheduler=LambdaLR(optimizer, lambda ep: 1 / (1 + 0.05*ep) ))
    clip_callback = GradientClipCallback(clip_type='value', clip_value=2)
    save_dir = os.path.join(
        root_path, 
        f'model/{args.data_type}', 
        f'fold{args.fold}'
    )
    save_callback = SaveModelCallback(
        top=1, 
        save_dir=save_dir
    )
    if args.cv:
        callbacks = [
            lrschedule_callback,
            clip_callback,
            save_callback,
        ]
    else:
        callbacks = [
            lrschedule_callback,
            clip_callback,
            save_callback,
        ]
    # callbacks.append(Unfreeze_Callback(embedding_param ,args.fix_embed_epoch))

    if args.use_bert:
        if args.fix_bert_epoch != 0:
            callbacks.append(Unfreeze_Callback(model.lattice_embed, args.fix_bert_epoch))
        else:
            bert_embedding.requires_grad = True

    callbacks.append(EarlyStopCallback(args.early_stop))

    if args.warmup > 0 and args.model == 'transformer':
        callbacks.append(WarmupCallback(warmup=args.warmup,))
    return callbacks


def write_pred_tsv(pred, ):
    pred = [int(ele) for sublist in pred for ele in sublist]

    dev_data = load_dev()
    origin_data = load_dev(simplify=False)

    offset_map = []
    for idx in range(len(dev_data)):
        dev_data[idx], map_arr = romove_redundant_str(dev_data[idx], dev_mode=True)
        offset_map.append(map_arr)

    pred = [vocabs['label'].to_word(ele) for ele in pred]

    pred_per_article = split_to_pred_per_article([pred], count_article_length(dev_data))
    print('writing output.tsv file...')
    
    write_result(dev_data, pred_per_article, offset_map, origin_data, output_path='../pred/output.tsv')


def visualize_error(ds, target, pred):
    chars = list(ds['raw_chars'])
    chars = [''.join(sublist) for sublist in chars]
    output = ''

    for i, sublist in enumerate(target):
        for j, ele in enumerate(sublist):
            line = f'{chars[i][j]} {ele} {pred[i][j]}\n'
            output+=line

    with open('../visualize.txt', 'w', encoding='utf-8') as f:
        print('Visualize validation....')
        f.write(output)


print('label num:', len(vocabs['label']))
if args.status == 'train':
    trainer = Trainer(
        datasets['train'],model,optimizer,loss,args.batch,
        n_epochs=args.epoch,
        dev_data=datasets['dev'],
        metrics=metrics,
        device=device,
        callbacks=create_cb(),
        dev_batch_size=args.test_batch,
        test_use_tqdm=False,
        check_code_level=-1,
        update_every=args.update_every
    )
    trainer.train()
    print('Evaluating...')
    with torch.no_grad():
        model = Predictor(model)
        pred = model.predict(
            datasets['dev'],
            seq_len_field_name='seq_len',
        )['pred']
    pred = [[vocabs['label'].to_word(ele) for ele in arr] for arr in pred]
    target = list(datasets['dev']['target'])
    target = [[vocabs['label'].to_word(ele) for ele in arr] for arr in target]
    cls_res = classification_report(target, pred)

    print(cls_res)
    print('=============================')
    visualize_error(datasets['dev'], target, pred)
    # Prediction to aicup data
    if args.do_pred:
        print('predicting...')
        pred = model.predict(
            datasets['aicup_dev'],
            seq_len_field_name='seq_len',)['pred']
        convert_pred_and_write(
            pred,
            f'../pred/{args.pred_dir}/{args.fold}.npy', 
            vocabs['label']
        )
        write_pred_tsv(pred)
else:
    models_path = [
        '/home/dy/flat-chinese-ner/model/default/fold0/2020-12-21-08-46-26/epoch-11_step-2002_f-0.769890.pt',
        '/home/dy/flat-chinese-ner/model/default/fold1/2020-12-21-09-02-29/epoch-15_step-2730_f-0.776722.pt',
        '/home/dy/flat-chinese-ner/model/default/fold2/2020-12-21-09-18-33/epoch-10_step-1810_f-0.769046.pt',
        '/home/dy/flat-chinese-ner/model/default/fold3/2020-12-21-09-32-06/epoch-8_step-1456_f-0.788785.pt',
        '/home/dy/flat-chinese-ner/model/default/fold4/2020-12-21-09-48-03/epoch-14_step-2548_f-0.769837.pt',
    ]
    mpath = models_path[args.fold]
    print('predicting...')
    model = Predictor(torch.load(mpath, map_location=device))
    pred = model.predict(
        datasets['aicup_dev'],
        seq_len_field_name='seq_len',
    )
    pred = pred['pred']

    pred_num = [int(ele) for sublist in pred for ele in sublist]
    with open(f'./pred/pred{args.fold}.npy', 'wb') as f:
                print(f'writing pred{args.fold}.npy...')    
                np.save(f, np.array(pred_num))
    write_pred_tsv(pred)