import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizer
from keras.preprocessing.sequence import pad_sequences
import torch.nn.functional as F
from modelvision import Stem, Inception_ResNet_A, Reduction_A, Inception_ResNet_B, Reduciton_B, Inception_ResNet_C, \
    Conv2d


class Inception_ResNetv2(nn.Module):
    def __init__(self, in_channels=3, classes=8, k=256, l=256, m=384, n=384):
        super(Inception_ResNetv2, self).__init__()
        blocks = []
        blocks.append(Stem(in_channels))
        for i in range(10):
            blocks.append(Inception_ResNet_A(320, 0.17))
        blocks.append(Reduction_A(320, k, l, m, n))
        for i in range(20):
            blocks.append(Inception_ResNet_B(1088, 0.10))
        blocks.append(Reduciton_B(1088))
        for i in range(9):
            blocks.append(Inception_ResNet_C(2080, 0.20))
        blocks.append(Inception_ResNet_C(2080, activation=False))
        self.features = nn.Sequential(*blocks)
        self.conv = Conv2d(2080, 1536, 1, stride=1, padding=0, bias=False)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.transformer_layer_left = nn.TransformerEncoderLayer(d_model=1536, nhead=16)
        self.transformer_layer_right = nn.TransformerEncoderLayer(d_model=1536, nhead=16)

        self.transformer_left = nn.TransformerEncoder(self.transformer_layer_left, num_layers=2)
        self.transformer_right = nn.TransformerEncoder(self.transformer_layer_right, num_layers=2)

        self.ln_left = nn.LayerNorm(1536)
        self.ln_right = nn.LayerNorm(1536)

        self.classifier = nn.Sequential(
            nn.Linear(1536 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 126),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(126, classes)
        )

    def forward(self, left_images, right_images):
        # 图像编码
        left_images_features = self.features(left_images)
        right_images_features = self.features(right_images)

        left_images_features = self.conv(left_images_features)
        right_images_features = self.conv(right_images_features)

        left_images_features = self.global_average_pooling(left_images_features)
        right_images_features = self.global_average_pooling(right_images_features)
        left_images_features = left_images_features.view(left_images_features.size(0), -1)
        right_images_features = right_images_features.view(right_images_features.size(0), -1)

        left_transformer_output = self.transformer_left(left_images_features)
        right_transformer_output = self.transformer_right(right_images_features)
        left_transformer_output = self.ln_left(left_transformer_output)
        right_transformer_output = self.ln_right(right_transformer_output)

        combined_features = torch.cat((left_transformer_output, right_transformer_output), dim=1)

        x = self.classifier(combined_features)

        return left_transformer_output, right_transformer_output


class Vision(nn.Module):
    def __init__(self):
        super(Vision, self).__init__()
        model = Inception_ResNetv2()
        checkpoint = torch.load("Inception_ResNetv2.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        # model = nn.Sequential(*list(model.children())[:-2])
        self.model = model

        for param in model.parameters():
            param.requires_grad = False

    def forward(self, left_images, right_images):
        left_images_features, right_images_features = self.model(left_images, right_images)

        return left_images_features, right_images_features


# input[batchsize, 300, 300]
class ModalClassifier(nn.Module):
    def __init__(self, classes=8):
        super(ModalClassifier, self).__init__()

        self.vision = Vision()

        self.linear_left_image = nn.Linear(1556, 768)
        self.linear_right_image = nn.Linear(1556, 768)

        self.attention_layer1_left = nn.MultiheadAttention(embed_dim=768, num_heads=16)
        self.attention_layer1_right = nn.MultiheadAttention(embed_dim=768, num_heads=16)
        self.attention_layer2_left = nn.MultiheadAttention(embed_dim=768, num_heads=16)
        self.attention_layer2_right = nn.MultiheadAttention(embed_dim=768, num_heads=16)

        self.transformer_layer_left = nn.TransformerEncoderLayer(d_model=768 * 2, nhead=16)
        self.transformer_layer_right = nn.TransformerEncoderLayer(d_model=768 * 2, nhead=16)

        self.transformer_layer1 = nn.TransformerEncoderLayer(d_model=768 * 4, nhead=16)
        self.transformer_layer2 = nn.TransformerEncoderLayer(d_model=768 * 4, nhead=16)

        self.transformer_left = nn.TransformerEncoder(self.transformer_layer_left, num_layers=2)
        self.transformer_right = nn.TransformerEncoder(self.transformer_layer_right, num_layers=2)

        self.transformer1 = nn.TransformerEncoder(self.transformer_layer1, num_layers=2)
        self.transformer2 = nn.TransformerEncoder(self.transformer_layer2, num_layers=2)

        # 添加LN层
        self.ln1_left = nn.LayerNorm(768)
        self.ln1_right = nn.LayerNorm(768)
        self.ln2_left = nn.LayerNorm(768)
        self.ln2_right = nn.LayerNorm(768)
        self.ln3_left = nn.LayerNorm(768 * 2)
        self.ln3_right = nn.LayerNorm(768 * 2)
        self.ln4 = nn.LayerNorm(768 * 4)
        self.ln5 = nn.LayerNorm(768 * 4)

        self.classifier = nn.Sequential(
            nn.Linear(768 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 126),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(126, classes)
        )

    def forward(self, left_images, right_images, ages, sexes, left_keywords, right_keywords):
        # 图像编码
        left_images_features, right_images_features = self.vision(left_images, right_images)

        # 将年龄和性别与图像特征融合
        ages_expanded = ages.unsqueeze(1).repeat(1, 10)
        sexes_expanded = sexes.unsqueeze(1).repeat(1, 10)
        ages_expanded = F.normalize(ages_expanded, p=2, dim=0)
        sexes_expanded = F.normalize(sexes_expanded, p=2, dim=0)

        combined_left_images_features = torch.cat((left_images_features,
                                                   ages_expanded, sexes_expanded), dim=1)
        combined_right_images_features = torch.cat((right_images_features,
                                                    ages_expanded, sexes_expanded), dim=1)

        # 使用线性层将图像特征映射到与文本特征相同的维度
        combined_left_images_features = self.linear_left_image(combined_left_images_features)
        combined_right_images_features = self.linear_right_image(combined_right_images_features)
        # [seq_len, batch_size, embed_dim]大小输入[1, batchsize, 768]
        combined_left_images_features = combined_left_images_features.unsqueeze(0)
        combined_right_images_features = combined_right_images_features.unsqueeze(0)
        left_keywords_encoding = left_keywords.unsqueeze(0)
        right_keywords_encoding = right_keywords.unsqueeze(0)

        # 图片编码和文本编码进行双向交叉注意力融合
        left_attention_output1, _ = self.attention_layer1_left(combined_left_images_features,
                                                               combined_left_images_features,
                                                               left_keywords_encoding)
        right_attention_output1, _ = self.attention_layer1_right(combined_right_images_features,
                                                                 combined_right_images_features,
                                                                 right_keywords_encoding)
        left_attention_output1 = self.ln1_left(left_attention_output1)
        right_attention_output1 = self.ln1_right(right_attention_output1)
        left_attention_output2, _ = self.attention_layer2_left(left_keywords_encoding,
                                                               combined_left_images_features,
                                                               combined_left_images_features)
        right_attention_output2, _ = self.attention_layer2_right(right_keywords_encoding,
                                                                 combined_right_images_features,
                                                                 combined_right_images_features)
        left_attention_output2 = self.ln2_left(left_attention_output2)
        right_attention_output2 = self.ln2_right(right_attention_output2)

        # 压缩回[batchsize, 768]，拼贴
        left_attention_output1 = left_attention_output1.squeeze(0)
        right_attention_output1 = right_attention_output1.squeeze(0)
        left_attention_output2 = left_attention_output2.squeeze(0)
        right_attention_output2 = right_attention_output2.squeeze(0)
        combined_features_left = torch.cat((left_attention_output1, left_attention_output2), dim=1)
        combined_features_right = torch.cat((right_attention_output1, right_attention_output2), dim=1)
        combined_features_left = combined_features_left.unsqueeze(0)
        combined_features_right = combined_features_right.unsqueeze(0)

        # Transformer层
        left_transformer_output = self.transformer_left(combined_features_left)
        right_transformer_output = self.transformer_right(combined_features_right)
        left_transformer_output = self.ln3_left(left_transformer_output)
        right_transformer_output = self.ln3_right(right_transformer_output)

        # 拼贴
        left_transformer_output = left_transformer_output.squeeze(0)
        right_transformer_output = right_transformer_output.squeeze(0)
        combined_features = torch.cat((left_transformer_output, right_transformer_output), dim=1)
        combined_features = combined_features.unsqueeze(0)

        # Transformer层
        transformer_output1 = self.transformer1(combined_features)
        transformer_output1 = self.ln4(transformer_output1)
        transformer_output2 = self.transformer2(transformer_output1)
        transformer_output2 = self.ln5(transformer_output2)
        transformer_output = transformer_output2.squeeze(0)

        x = self.classifier(transformer_output)

        return x


# 测试函数
if __name__ == '__main__':
    # 创建模型实例
    model = ModalClassifier()
    # 生成随机数据来模拟输入
    batch_size = 8
    ages = torch.randint(low=40, high=91, size=(batch_size,), dtype=torch.float32)
    sexes = torch.randint(0, 2, (batch_size,), dtype=torch.float32)
    left_images = torch.randn(batch_size, 3, 300, 300)
    right_images = torch.randn(batch_size, 3, 300, 300)
    left_keywords = ['post retinal laser surgery，moderate non proliferative retinopathy', 'normal fundus',
                     'normal fundus', 'normal fundus', 'mild nonproliferative retinopathy', 'cataract',
                     'vitreous degeneration，lens dust', 'normal fundus']
    right_keywords = ['post retinal laser surgery，moderate non proliferative retinopathy', 'normal fundus',
                      'mild nonproliferative retinopathy', 'moderate non proliferative retinopathy',
                      'mild nonproliferative retinopathy', 'drusen', 'lens dust，normal fundus',
                      'branch retinal artery occlusion']

    # 对左眼和右眼的诊断关键字进行token编码
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased')
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

    # 将模型移动到CUDA设备
    device = torch.device("cuda")
    model.to(device)
    ages = ages.to(device)
    sexes = sexes.to(device)
    left_images = left_images.to(device)
    right_images = right_images.to(device)
    left_keywords_encoding = left_keywords_encoding.to(device)
    right_keywords_encoding = right_keywords_encoding.to(device)

    # 测试模型的前向传播
    outputs = model(left_images, right_images, ages, sexes, left_keywords_encoding, right_keywords_encoding)
    print("Success! Output shape [batchsize, class]:", outputs)

    # # 创建模型实例
    # model = Vision()
    # # 生成随机数据来模拟输入
    # batch_size = 4
    # left_images = torch.randn(batch_size, 3, 300, 300)
    # right_images = torch.randn(batch_size, 3, 300, 300)
    #
    # # 将模型移动到CUDA设备
    # device = torch.device("cuda")
    # model.to(device)
    # left_images = left_images.to(device)
    # right_images = right_images.to(device)
    #
    # # 测试模型的前向传播
    # outputs = model(left_images, right_images)
    # print(outputs[0].size(), outputs[1].size())
    # # print("Success! Output shape [batchsize, class]:", outputs.size())
    # print(outputs)
