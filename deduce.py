from transformers import BertModel, BertTokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
from torchvision import transforms
from PIL import Image
from model import ModalClassifier


def custom_collate_fn(left_keywords, right_keywords):
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

    return left_keywords_encoding, right_keywords_encoding


image_folder = "D:\\Documents\\Datasets\\eyes\\ODIR-5K\\ODIR-5K_Training_Images"

transform = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(300),
    transforms.ToTensor()
])

# 加载模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')
device = torch.device('cuda')
print(f'使用设备: {device}')
model_name = "ModalClassifier"
model = ModalClassifier().to(device)
checkpoint_path = f"{model_name}.pth"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

while True:

    while True:
        # 输入ID
        ID = input("请输入患者ID（用于获取图片地址）：")
        try:
            ID = int(ID)
            if 0 <= ID <= 5000:
                break
            else:
                print("ID不存在，请重新输入。")
        except ValueError:
            print("ID不存在，请重新输入。")

    while True:
        # 输入年龄
        age = input("请输入患者年龄：")
        try:
            age = int(age)
            if 0 < age <= 150:
                break
            else:
                print("年龄必须是0到150之间的正整数，请重新输入。")
        except ValueError:
            print("年龄必须是0到150之间的正整数，请重新输入。")

    while True:
        # 输入性别
        gender = input("请输入患者性别（男/女）：")
        if gender == '男':
            sex = 0
            break
        if gender == '女':
            sex = 1
            break
        print("输入有误，请重新输入。")

    while True:
        # 输入文本
        left_keywords = input("请输入患者左眼诊断信息（英文）：")
        words = left_keywords.split('，')
        if (any(any(c.isalpha() for c in word) for word in words) and
                all(all(ord(c) < 128 and (c.isalpha() or c == ' ' or c == '-') for c in word) for word in words)):
            # 输入为至少包含一个英文字符并且是英文字符串或用逗号和空格分割的英文字符串
            break
        print("输入至少包含一个英文字符，关键词之间用“，”隔开，请重新输入。")

    while True:
        # 输入文本
        right_keywords = input("请输入患者右眼诊断信息（英文）：")
        words = right_keywords.split('，')
        if (any(any(c.isalpha() for c in word) for word in words) and
                all(all(ord(c) < 128 and (c.isalpha() or c == ' ' or c == '-') for c in word) for word in words)):
            # 输入为至少包含一个英文字符并且是英文字符串或用逗号和空格分割的英文字符串
            break
        print("输入至少包含一个英文字符，关键词之间用“，”隔开，请重新输入。")

    print(f"患者ID：{ID}，年龄：{age}，性别：{gender}，左眼症状：{left_keywords}，右眼症状：{right_keywords}")
    age = torch.tensor(age, dtype=torch.float32)
    sex = torch.tensor(sex, dtype=torch.float32)
    left_keywords = [f'{left_keywords}']
    right_keywords = [f'{right_keywords}']
    left_keywords_encoding, right_keywords_encoding = custom_collate_fn(left_keywords, right_keywords)

    # 模型推理
    with torch.no_grad():

        # 构建图片路径
        image_left_path = f"{image_folder}/{ID}_left.jpg"
        image_right_path = f"{image_folder}/{ID}_right.jpg"

        # 读取图片
        image_left = transform(Image.open(image_left_path).convert("RGB"))
        image_right = transform(Image.open(image_right_path).convert("RGB"))

        # 添加批处理维度
        image_left = image_left.unsqueeze(0).to(device)
        image_right = image_right.unsqueeze(0).to(device)
        age = age.unsqueeze(0).to(device)
        sex = sex.unsqueeze(0).to(device)
        left_keywords_encoding = left_keywords_encoding.to(device)
        right_keywords_encoding = right_keywords_encoding.to(device)

        outputs = model(image_left, image_right, age, sex, left_keywords_encoding, right_keywords_encoding)
        preds = outputs.sigmoid().round().detach().cpu().numpy()

        # print(outputs)
        # print(preds)

        # 定义疾病名称和对应的概率
        disease_labels = ['N', '糖尿病视网膜病变', '青光眼', '白内障', '老年性黄斑病变', '高血压', '近视性视网膜病变', '其他严重眼病/视网膜病变']
        probs = outputs.sigmoid().cpu().detach().numpy()[0]

        # 获取预测的疾病名称和概率
        pred_labels = [disease_labels[i] for i in range(len(preds[0])) if preds[0][i] == 1]
        pred_probs = [probs[i] for i in range(len(preds[0])) if preds[0][i] == 1]

        # 获取最大概率对应的索引
        max_prob_index = torch.argmax(outputs, dim=1)

        # 如果最大概率对应的索引是0（即正常），则直接打印“该患者正常”
        if max_prob_index == 0:
            output_str = "该名患者眼部正常。"
        else:
            output_str = "该名患者可能存在以下疾病："
            for label, prob in zip(pred_labels, pred_probs):
                output_str += f"{label}({prob * 100:.2f}%) "

        print(output_str)


