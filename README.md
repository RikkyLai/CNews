# CNews Note （Pytorch）中文文本分类

数据集：cnews中文文本分类，由清华大学根据新浪新闻RSS订阅频道2005-2011年间的历史数据筛选过滤生成

训练集50000、验证集5000、测试集10000、词汇5000。

包含10个分类，“体育”、“财经”、“房产”、“家居”、“教育”、“科技”、“时尚”、“时政”、“娱乐”

目录结构
---

目前只有RNN模型

```
CNEWS
│  Cnews_classifier.ipynb
│  cnews_loader.py
│  model.py
│  ReadMe.md
│  train.py
|  requirement.txt
├─.idea     
├─.ipynb_checkpoints     
├─dataset
│      cnews.test.txt
│      cnews.train.txt
│      cnews.val.txt
│      cnews.vocab.txt      
└─__pycache__
```

*cnews_loader.py* 加载文本

*model.py*  模型

*train.py* 训练

*Cnews_classifier.ipynb* 这个涵盖了所有函数，包括预测，还有整个流程。




