import numpy as np
import os
import tensorflow.contrib.keras as kr
import torch


# 读取词汇表
def read_vocab(vocab_dir):
    with open(vocab_dir, 'r', encoding='utf-8', errors='ignore') as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


# 读取分类目录，固定
def read_category():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    categories = [x for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


# 将文件转换为id表示
def process_file(filename, word_to_id, cat_to_id, max_length=600):
    contents, labels = [], []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])  # 将每句话id化
        label_id.append(cat_to_id[labels[i]])  # 每句话对应的类别的id

    # # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    x_pad = torch.LongTensor(x_pad)
    # y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, torch.LongTensor(label_id)


from torch.utils.data import Dataset


class textData(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """
    def __init__(self, train=False, val=False):
        categories, cat_to_id = read_category()
        words, word_to_id = read_vocab('./dataset/cnews.vocab.txt')
        if train:
            # 数据加载及分批
            # 获取训练数据每个字的id和对应标签的one-hot形式
            self.data, self.label = process_file('./dataset/cnews.train.txt', word_to_id, cat_to_id, 600)
        if val:
            self.data, self.label = process_file('./dataset/cnews.val.txt', word_to_id, cat_to_id, 600)
        if not train and not val:
            self.data, self.label = process_file('./dataset/cnews.test.txt', word_to_id, cat_to_id, 600)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]
