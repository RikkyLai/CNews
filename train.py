import torch.optim as optim
import time
import torch
from model import TextRNN
from cnews_loader import textData
from torch import nn
from torch.utils.data import DataLoader
from sklearn import metrics
import numpy as np
import torch.nn.functional as F


def evaluate(model, data_loader, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    for i, data in enumerate(val_loader):
        x, y = data
        texts, labels = x.to(device), y.to(device)
        with torch.no_grad():
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        # report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_loader.dataset), confusion
    return acc, loss_total / len(data_loader.dataset)


EPOCH = 30
batch_size = 32
best_epoch, best_acc = 0, 0
#保存训练模型
file_name = 'cnews_best.pt'
train_data = textData(train=True)
val_data = textData(val=True)
test_data = textData()
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
model = TextRNN()

# 损失函数:这里用交叉熵
criterion = nn.CrossEntropyLoss()
# 优化器 这里用SGD
optimizer = optim.Adam(model.parameters(), lr=0.001)

# device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练
for epoch in range(EPOCH):
    start_time = time.time()
    for i, data in enumerate(train_loader):
        model.train()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # 前向传播
        outputs = model(inputs)
        # 计算损失函数
        loss = criterion(outputs, labels)
        # 清空上一轮梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
        accuracy = torch.mean((torch.argmax(outputs, 1) == labels.data).float())
        print('epoch{} loss:{:.4f} acc:{:.4f} time:{:.4f}'.format(epoch+1, loss.item(), accuracy.item(), time.time()-start_time))
    if epoch % 1 == 0:
        val_acc, val_loss = evaluate(model, val_loader)
        print('epoch{} val_loss:{:.4f} val_acc:{:.4f}'.format(epoch + 1, val_loss, val_acc))
        if val_acc > best_acc:
            best_epoch = epoch
            best_acc = val_acc
            torch.save(model.state_dict(), file_name)
        print('best acc:', best_acc, 'best epoch:', best_epoch)





