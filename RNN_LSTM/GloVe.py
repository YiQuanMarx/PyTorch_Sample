from torchnlp.word_to_vector import GloVe
# import torch
# Tip 2:GloVe
vectors = GloVe()
print(vectors['hello'])
# print(torch.__version__)

'''
Tip 2:Glove

在使用预训练的词向量模型（如 GloVe）时，词向量的数值不是随机的，而是在大规模文本语料库上通过训练得到的。

GloVe（Global Vectors for Word Representation）是一种基于全局词频统计的词嵌入模型，它在训练过程中利用了大量的文本数据。因此，GloVe 提供的词向量是经过训练的，并且可以在许多自然语言处理任务中发挥很好的作用。

在你的代码中，通过 `vectors = GloVe()` 获取到了预训练的 GloVe 词向量模型，然后使用 `vectors['hello']` 可以获取单词 "hello" 对应的预训练词向量。

这些词向量的数值反映了在训练数据中单词的语义关系。例如，相似的单词在词向量空间中会有相近的表示。

需要注意的是，预训练的词向量模型通常基于大规模的文本数据训练得到，所以它们通常能够捕捉到很多通用的语义信息。
'''