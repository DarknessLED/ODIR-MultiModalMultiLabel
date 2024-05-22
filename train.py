import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from keras.src.utils import pad_sequences
from sklearn.exceptions import UndefinedMetricWarning
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import BertTokenizer, BertModel
from dataset import ODIR5KDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import ModalClassifier

# 数据转换
channel_mean = torch.Tensor([0.300, 0.190, 0.103])
channel_std = torch.Tensor([0.310, 0.212, 0.137])
# channel_mean = torch.Tensor([0.485, 0.456, 0.406])
# channel_std = torch.Tensor([0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize(320),
    transforms.RandomRotation(degrees=90),
    transforms.CenterCrop(300),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=channel_mean, std=channel_std)
])

# 加载数据集
dataset = ODIR5KDataset(
    csv_file="D:\\Documents\\Datasets\\eyes\\ODIR-5K\\ODIR-5K_Training_Annotations.csv",
    image_dir="D:\\Documents\\Datasets\\eyes\\ODIR-5K\\ODIR-5K_Training_Images\\",
    transform=transform_train
)

# 划分训练集和验证集
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


def custom_collate_fn(batch):
    # 解包批次中的数据
    left_images, right_images, ages, sexes, left_keywords, right_keywords, labels = zip(*batch)

    # 将图像、年龄、性别和标签堆叠起来
    left_images = torch.stack(left_images, dim=0)
    right_images = torch.stack(right_images, dim=0)
    ages = torch.stack(ages, dim=0)
    sexes = torch.stack(sexes, dim=0)
    labels = torch.stack(labels, dim=0)

    # 将所有样本的left_keywords和right_keywords合并为一个列表
    left_keywords = [keyword for sublist in left_keywords for keyword in sublist]
    right_keywords = [keyword for sublist in right_keywords for keyword in sublist]

    # 对左眼和右眼的诊断关键字进行token编码
    left_keyword_ids = [tokenizer.encode(keyword, add_special_tokens=True) for keyword in left_keywords]
    right_keyword_ids = [tokenizer.encode(keyword, add_special_tokens=True) for keyword in right_keywords]
    # 设置为20以确保所有序列填充到相同的长度
    max_seq_length = 20
    left_keywords_ids = pad_sequences(left_keyword_ids, maxlen=max_seq_length, padding='post', value=0)
    right_keywords_ids = pad_sequences(right_keyword_ids, maxlen=max_seq_length, padding='post', value=0)
    # 将编码结果转换为Tensor
    left_keywords_encoding = torch.tensor(left_keywords_ids)
    right_keywords_encoding = torch.tensor(right_keywords_ids)
    # bert特征编码[CLS]标记
    left_keywords_encoding = bert(left_keywords_encoding).pooler_output
    right_keywords_encoding = bert(right_keywords_encoding).pooler_output

    return left_images, right_images, ages, sexes, left_keywords_encoding, right_keywords_encoding, labels


def compute_metrics(y_true, y_pred):
    # 计算每个标签的指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    # 计算混淆矩阵
    # confusion_matrices = multilabel_confusion_matrix(y_true.view(-1).numpy(), y_pred.view(-1).numpy())
    return accuracy, precision, recall, f1


def compute_loss_weights(labels):
    num_labels = labels.size(1)
    pos_weight = []

    for i in range(num_labels):
        positives = labels[:, i].sum().item()
        negatives = labels.size(0) - positives

        # if positives > 0:
        #     weight = (negatives / positives)
        # else:
        #     weight = (negatives / 0.8)
        #
        # weight = weight ** 0.5
        # pos_weight.append(weight)

        if positives > 0:
            pos_ratio = positives / (positives + negatives)
        else:
            pos_ratio = 0.5  # 如果没有正样本，则将正样本比例设置为0.5，即正负样本平衡

        weight = 1 / (pos_ratio + 1e-6)  # 避免除零错误
        weight = weight ** 0.7
        pos_weight.append(weight)

    # print(torch.tensor(pos_weight))

    return torch.tensor(pos_weight)


# 创建数据加载器
BATCH_SIZE = 8
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

# 超参数、初始化模型、损失函数和优化器
EPOCHS = 50
LEARNING_RATE = 0.0001
device = torch.device('cuda')
print(f'使用设备: {device}')
model_name = "ModalClassifier"
model = globals()[model_name]().to(device)
# criterion = nn.BCEWithLogitsLoss()
optimizer = optim.RAdam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.002)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')
# 设置警告过滤器
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

# 检查点路径
checkpoint_path = f"{model_name}.pth"

# 加载检查点
train_loss_history = []
train_total_loss_history = []
valid_loss_history = []

train_acc_history = []
valid_acc_history = []
train_pre_history = []
valid_pre_history = []
train_rec_history = []
valid_rec_history = []
train_f1_history = []
valid_f1_history = []


if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss_history = checkpoint['train_loss_history']
    train_total_loss_history = checkpoint['train_total_loss_history']
    valid_loss_history = checkpoint['valid_loss_history']
    train_acc_history = checkpoint['train_acc_history']
    valid_acc_history = checkpoint['valid_acc_history']
    train_pre_history = checkpoint['train_pre_history']
    valid_pre_history = checkpoint['valid_pre_history']
    train_rec_history = checkpoint['train_rec_history']
    valid_rec_history = checkpoint['valid_rec_history']
    train_f1_history = checkpoint['train_f1_history']
    valid_f1_history = checkpoint['valid_f1_history']

else:
    # epoch = checkpoint['epoch']
    epoch = 0

print(f"START Epoch: {epoch + 1:2d}")
k = 0
# 训练模型
for epoch in range(epoch, EPOCHS):
    model.train()
    all_preds = []
    all_labels = []
    train_loss = 0.0
    for step, (
            images_left, images_right, ages, sexes, left_keywords_encoding, right_keywords_encoding,
            labels) in enumerate(
        train_loader,
        start=1):
        # print("Left image shape:", images_left.shape)
        # print("Right image shape:", images_right.shape)
        # print("Age:", ages)
        # print("Sex:", sexes)
        # print("Left keywords encoding:", left_keywords)
        # print("Right keywords encoding:", right_keywords)
        # print("Label:", labels)

        images_left = images_left.to(device)
        images_right = images_right.to(device)
        ages = ages.to(device)
        sexes = sexes.to(device)
        left_keywords_encoding = left_keywords_encoding.to(device)
        right_keywords_encoding = right_keywords_encoding.to(device)
        labels = labels.to(device)

        try:
            outputs = model(images_left, images_right, ages, sexes, left_keywords_encoding, right_keywords_encoding)
        except AssertionError as e:
            print(f"AssertionError occurred in step {step}: {e}")
            continue
        optimizer.zero_grad()
        k += 1
        # preds = torch.sigmoid(outputs)
        pos_weight = compute_loss_weights(labels).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss

        # preds = (outputs.sigmoid() > 0.5).float().detach().cpu().numpy()

        preds = outputs.sigmoid().round().detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        # print(f"preds:{preds}")
        # print(f"labels:{labels}")

        train_accuracy, train_precision, train_recall, train_f1 = compute_metrics(labels, preds)

        # 捕获警告并打印信息
        with warnings.catch_warnings(record=True) as w:
            try:
                train_accuracy, train_precision, train_recall, train_f1 = compute_metrics(labels, preds)
                print(f"Step: {step}, Loss: {loss}")
                if step % 20 == 0:
                    print(f"outputs:{outputs}")
                    print(f"preds:{preds}")
                    print(f"labels:{labels}")
                    print(
                        f"Accuracy: {train_accuracy}, Precision: {train_precision}, Recall: {train_recall}, F1 Score: {train_f1}")

            except UndefinedMetricWarning as e:
                train_accuracy = accuracy_score(labels, preds)
                print(f"Step: {step}, Loss: {loss}")
                if step % 5 == 0:
                    print(f"Accuracy: {train_accuracy}")

        train_loss_history.append(loss)
    train_accuracy, train_precision, train_recall, train_f1 = compute_metrics(all_labels, all_preds)
    print(
        f"Total Loss: {train_loss}, Accuracy: {train_accuracy}, Precision: {train_precision},"
        f"Recall: {train_recall}, F1 Score: {train_f1}"
    )

    train_total_loss_history.append(train_loss)
    train_acc_history.append(train_accuracy)
    train_pre_history.append(train_precision)
    train_rec_history.append(train_recall)
    train_f1_history.append(train_f1)

    # 验证模型
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        valid_loss = 0.0
        for images_left, images_right, ages, sexes, left_keywords_encoding, right_keywords_encoding, labels in val_loader:
            images_left = images_left.to(device)
            images_right = images_right.to(device)
            ages = ages.to(device)
            sexes = sexes.to(device)
            left_keywords_encoding = left_keywords_encoding.to(device)
            right_keywords_encoding = right_keywords_encoding.to(device)
            labels = labels.to(device)

            try:
                outputs = model(images_left, images_right, ages, sexes, left_keywords_encoding, right_keywords_encoding)
            except AssertionError as e:
                print(f"AssertionError : {e}")
                continue

            pos_weight = compute_loss_weights(labels).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = criterion(outputs, labels).item()

            # preds = (outputs.sigmoid() > 0.5).float().detach().cpu().numpy()

            preds = outputs.sigmoid().round().detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

            valid_loss += loss

        valid_accuracy, valid_precision, valid_recall, valid_f1 = compute_metrics(all_labels, all_preds)

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {valid_loss}")
        print(f"Accuracy: {valid_accuracy}, Precision: {valid_precision}, Recall: {valid_recall}, F1 Score: {valid_f1}")

        # 保存验证集指标
        valid_loss_history.append(valid_loss)
        valid_acc_history.append(valid_accuracy)
        valid_pre_history.append(valid_precision)
        valid_rec_history.append(valid_recall)
        valid_f1_history.append(valid_f1)

        # 保存模型检查点
        if valid_loss <= min(valid_loss_history):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_history': train_loss_history,
                'train_total_loss_history': train_total_loss_history,
                'valid_loss_history': valid_loss_history,
                'train_acc_history': train_acc_history,
                'valid_acc_history': valid_acc_history,
                'train_pre_history': train_pre_history,
                'valid_pre_history': valid_pre_history,
                'train_rec_history': train_rec_history,
                'valid_rec_history': valid_rec_history,
                'train_f1_history': train_f1_history,
                'valid_f1_history': valid_f1_history,
            }, checkpoint_path)

# # Loss 曲线
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 3, 1)
# plt.plot(train_total_loss_history, label='Train Loss', color='blue')
# plt.plot(valid_loss_history, label='Valid Loss', color='red')
# plt.xlabel('Step')
# plt.ylabel('Loss')
# plt.legend()
#
# # Loss 曲线
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 3, 2)
# plt.plot(train_loss_history, label='Train Step Loss', color='blue')
# plt.xlabel('Step')
# plt.ylabel('Loss')
# plt.legend()
#
# # Accuracy 曲线
# plt.subplot(2, 3, 3)
# plt.plot(train_acc_history, label='Train Accuracy', color='blue')
# plt.plot(valid_acc_history, label='Valid Accuracy', color='red')
# plt.xlabel('Step')
# plt.ylabel('Accuracy')
# plt.legend()
#
# # Precision 曲线
# plt.subplot(2, 3, 4)
# plt.plot(train_pre_history, label='Train Precision', color='blue')
# plt.plot(valid_pre_history, label='Valid Precision', color='red')
# plt.xlabel('Step')
# plt.ylabel('Precision')
# plt.legend()
#
# # Recall 曲线
# plt.subplot(2, 3, 5)
# plt.plot(train_rec_history, label='Train Recall', color='blue')
# plt.plot(valid_rec_history, label='Valid Recall', color='red')
# plt.xlabel('Step')
# plt.ylabel('Recall')
# plt.legend()
#
# # F1 Score 曲线
# plt.subplot(2, 3, 6)
# plt.plot(train_f1_history, label='Train F1 Score', color='blue')
# plt.plot(valid_f1_history, label='Valid F1 Score', color='red')
# plt.xlabel('Step')
# plt.ylabel('F1 Score')
# plt.legend()
#
# plt.tight_layout()
# plt.show()
