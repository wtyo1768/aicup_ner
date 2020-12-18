import jieba


import jieba.posseg as pseg


words = pseg.cut("我爱北京天安门")

text, a = list(words)
print(text)