import torch
import torch.nn as nn

word_to_ix={"hello":0,"world":1}

lookup_tensor=torch.tensor([word_to_ix["hello"]],dtype=torch.long)

# Tip 1:Emedding
embeds=nn.Embedding(2,5) # 2 words in vocab, 5 dimensional embeddings
hello_embed=embeds(lookup_tensor)
print(hello_embed)

"""
Tip 1:Embedding
embeds 是一个嵌入层，它被初始化为一个 2x5 的矩阵，这表示我们有两个单词（vocab 大小为2）和每个单词对应一个 5 维的嵌入向量。
hello_embed 中的向量值是由 nn.Embedding 层内部的权重矩阵所决定的，它们与具体的单词（"hello" 或 "world"）无关。
数值也都是随机分配的
"""