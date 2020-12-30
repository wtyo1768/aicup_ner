import jieba
import jieba.posseg as pseg



sentences = '你今天是不是忘記吃藥'
pos = pseg.cut(sentences)
pos_tag = []
for words, tag in pos:
    print(words, tag)