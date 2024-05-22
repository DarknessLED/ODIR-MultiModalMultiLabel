import pandas as pd
import torch
from PIL import Image
from keras.src.utils import pad_sequences
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor
from transformers import BertTokenizer, BertModel


class ODIR5KDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform):
        # 读取CSV文件
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        # 返回数据集的长度
        return len(self.annotations)

    def __getitem__(self, index):
        # 根据索引读取数据
        row = self.annotations.iloc[index]
        # 读取左右眼底图像
        left_image_path = self.image_dir + row['Left-Fundus']
        right_image_path = self.image_dir + row['Right-Fundus']
        left_image = Image.open(left_image_path).convert("RGB")
        right_image = Image.open(right_image_path).convert("RGB")

        # 读取参数指标
        age = torch.tensor(float(row['Patient Age']), dtype=torch.float32)
        sex = torch.tensor(int(row['Patient Sex'] == 'Female'), dtype=torch.float32)

        # 读取文本数据
        left_keywords = row['Left-Diagnostic Keywords'].split(',')
        right_keywords = row['Right-Diagnostic Keywords'].split(',')

        # 读取标签
        label = torch.tensor(row[['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']].astype(int).values, dtype=torch.float32)

        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        return left_image, right_image, age, sex, left_keywords, right_keywords, label


if __name__ == '__main__':
    # 示例CSV文件和图像文件夹路径
    csv_file = "D:\\Documents\\Datasets\\eyes\\ODIR-5K\\ODIR-5K_Training_Annotations.csv"
    image_dir = "D:\\Documents\\Datasets\\eyes\\ODIR-5K\\ODIR-5K_Training_Images\\"

    # # 创建数据集实例
    # dataset = ODIR5KDataset(csv_file=csv_file, image_dir=image_dir, transform=ToTensor())
    #
    # # 获取一个随机索引
    # random_index = random.randint(0, len(dataset) - 1)
    #
    # # 获取随机样本
    # sample = dataset[random_index]
    #
    # # 打印随机样本的索引和信息
    # print("Random sample index:", random_index + 2)
    # left_image, right_image, age, sex, left_keywords, right_keywords, label = sample
    # print("Left image shape:", left_image.shape)
    # print("Right image shape:", right_image.shape)
    # print("Age:", age)
    # print("Sex:", sex)
    # print("Left keywords:", left_keywords)
    # print("Right keywords:", right_keywords)
    # print("Label:", label)
    #
    # # 显示左右眼图片
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(left_image.permute(1, 2, 0))
    # plt.title('Left Eye Image')
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(right_image.permute(1, 2, 0))
    # plt.title('Right Eye Image')
    # plt.axis('off')
    #
    # plt.show()

    # 创建数据加载器
    BATCH_SIZE = 8
    # 数据转换
    channel_mean = torch.Tensor([0.300, 0.190, 0.103])
    channel_std = torch.Tensor([0.310, 0.212, 0.137])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=40),
        transforms.ToTensor(),
        transforms.Normalize(mean=channel_mean, std=channel_std),
    ])

    # 加载数据集
    dataset = ODIR5KDataset(
        csv_file="D:\\Documents\\Datasets\\eyes\\ODIR-5K\\ODIR-5K_Training_Annotations.csv",
        image_dir="D:\\Documents\\Datasets\\eyes\\ODIR-5K\\ODIR-5K_Training_Images\\",
        transform=transform
    )


    def custom_collate_fn(batch):
        # 解包批次中的数据
        left_images, right_images, ages, sexes, left_keywords, right_keywords, labels = zip(*batch)

        # 将图像、年龄、性别和标签堆叠起来
        left_images = torch.stack(left_images, dim=0)
        right_images = torch.stack(right_images, dim=0)
        ages = torch.stack(ages, dim=0)
        sexes = torch.stack(sexes, dim=0)
        labels = torch.stack(labels, dim=0)

        # # 将所有样本的left_keywords和right_keywords合并为一个列表
        left_keywords = [keyword for sublist in left_keywords for keyword in sublist]
        right_keywords = [keyword for sublist in right_keywords for keyword in sublist]

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert = BertModel.from_pretrained('bert-base-uncased')
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


    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

    # 获取一个随机批次的数据
    for batch in data_loader:
        left_images, right_images, ages, sexes, left_keywords_encoding, right_keywords_encoding, labels = batch
        print("Left image shape:", left_images.shape)
        print("Right image shape:", right_images.shape)
        print("Age:", ages)
        print("Sex:", sexes)
        print("Left keywords:", left_keywords_encoding)
        print("Right keywords:", right_keywords_encoding)
        print("Label:", labels)
        break  # 仅获取第一个批次的数据
