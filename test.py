import warnings
import torch
import torch.nn as nn
from keras.src.utils import pad_sequences
from sklearn.exceptions import UndefinedMetricWarning
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import BertTokenizer, BertModel
from model import ModalClassifier
from dataset import ODIR5KDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 数据转换
channel_mean = torch.Tensor([0.300, 0.190, 0.103])
channel_std = torch.Tensor([0.310, 0.212, 0.137])
transform = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(mean=channel_mean, std=channel_std)
])

# 加载数据集
dataset = ODIR5KDataset(
    csv_file="D:\\Documents\\Datasets\\eyes\\ODIR-5K\\ODIR-5K_Training_Annotations.csv",
    image_dir="D:\\Documents\\Datasets\\eyes\\ODIR-5K\\ODIR-5K_Training_Images\\",
    transform=transform
)

test_size = int(len(dataset) / 35)
test_dataset, remain_dataset = random_split(dataset, [test_size, len(dataset) - test_size])


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


# 创建数据加载器
BATCH_SIZE = 8
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

device = torch.device('cuda')
print(f'使用设备: {device}')
model_name = "ModalClassifier"
model = globals()[model_name]().to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')
checkpoint_path = f"{model_name}.pth"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
step = 0
model.eval()
with torch.no_grad():
    all_preds = []
    all_labels = []
    test_loss = 0.0
    for images_left, images_right, ages, sexes, left_keywords_encoding, right_keywords_encoding, labels in test_loader:

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

        # pos_weight = compute_loss_weights(labels).to(device)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(outputs, labels).item()

        preds = outputs.sigmoid().round().detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

        test_loss += loss

        step += 1
        print(f'step:{step}/13')

    test_accuracy, test_precision, test_recall, test_f1 = compute_metrics(all_labels, all_preds)

    print(f"Loss: {test_loss}")
    print(f"Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1 Score: {test_f1}")
