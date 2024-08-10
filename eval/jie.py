import jieba
w=jieba.cut("I am Lihua".strip().replace('。',''), cut_all=False)
p=' '.join(w)
print(p)
print(p.split())