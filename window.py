from tkinter import *
from tkinter import messagebox
from PIL import Image
from torchvision import transforms
import torch
from transformers import BertModel, BertTokenizer
from keras.preprocessing.sequence import pad_sequences

from model import ModalClassifier


def predict_disease(id, age, gender, left_keywords, right_keywords):
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

    age = torch.tensor(int(age), dtype=torch.float32)
    sex = torch.tensor(0 if gender == '男' else 1, dtype=torch.float32)
    left_keywords = [f'{left_keywords}']
    right_keywords = [f'{right_keywords}']
    left_keywords_encoding, right_keywords_encoding = custom_collate_fn(left_keywords, right_keywords)

    # 模型推理
    with torch.no_grad():

        # 构建图片路径
        image_left_path = f"{image_folder}/{id}_left.jpg"
        image_right_path = f"{image_folder}/{id}_right.jpg"

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

        # 定义疾病名称和对应的概率
        disease_labels = ['N', '糖尿病视网膜病变', '青光眼', '白内障', '老年性黄斑病变', '高血压', '近视性视网膜病变',
                          '其他严重眼病/视网膜病变']
        probs = outputs.sigmoid().cpu().detach().numpy()[0]

        # 获取预测的疾病名称和概率
        pred_labels = [disease_labels[i] for i in range(len(preds[0])) if preds[0][i] == 1]
        pred_probs = [probs[i] for i in range(len(preds[0])) if preds[0][i] == 1]

        # 获取最大概率对应的索引
        max_prob_index = torch.argmax(outputs, dim=1)

        # 如果最大概率对应的索引是0（即正常），则直接打印“该患者正常”，否则打印预测结果
        if max_prob_index == 0:
            output_str = "该名患者眼部正常。"
        else:
            output_str = "该名患者可能存在以下疾病："
            for label, prob in zip(pred_labels, pred_probs):
                output_str += f"{label}({prob * 100:.2f}%) "

        return output_str


def on_predict_button_clicked():
    # 获取输入的患者信息
    ID = id_entry.get()
    age = age_entry.get()
    gender = gender_entry.get()
    left_keywords = left_keywords_entry.get()
    right_keywords = right_keywords_entry.get()

    # 调用预测函数并显示结果
    result = predict_disease(ID, age, gender, left_keywords, right_keywords)
    messagebox.showinfo("预测结果", result)


# 创建主窗口
root = Tk()
root.title("眼部疾病预测系统")

# 创建标签和输入框
Label(root, text="患者ID：").grid(row=0, column=0)
id_entry = Entry(root)
id_entry.grid(row=0, column=1)

Label(root, text="年龄：").grid(row=1, column=0)
age_entry = Entry(root)
age_entry.grid(row=1, column=1)

Label(root, text="性别：").grid(row=2, column=0)
gender_entry = Entry(root)
gender_entry.grid(row=2, column=1)

Label(root, text="左眼症状：").grid(row=3, column=0)
left_keywords_entry = Entry(root)
left_keywords_entry.grid(row=3, column=1)

Label(root, text="右眼症状：").grid(row=4, column=0)
right_keywords_entry = Entry(root)
right_keywords_entry.grid(row=4, column=1)

# 创建预测按钮
predict_button = Button(root, text="预测", command=on_predict_button_clicked)
predict_button.grid(row=5, column=0, columnspan=2)

# 运行主循环
root.mainloop()
