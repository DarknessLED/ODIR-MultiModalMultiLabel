import numpy as np
import cv2
import os

mean, std = [0, 0, 0], [0, 0, 0]
img_list = []

imgs_path = "D:\\Documents\\Datasets\\eyes\\ODIR-5K\\ODIR-5K_Training_Images"
# imgs_path = "D:\\Documents\\Datasets\\dog-breed-identification\\test"
imgs_path_list = os.listdir(imgs_path)

len_ = len(imgs_path_list)
i = 0
batch = 100
step = int(len_ / batch)
print(step)

for j in range(step):
    batch_mean, batch_std = [], []
    start_idx = j * batch
    end_idx = min((j + 1) * batch, len(imgs_path_list))
    selected_imgs_path_list = imgs_path_list[start_idx:end_idx]

    for item in selected_imgs_path_list:
        img = cv2.imread(os.path.join(imgs_path, item))
        img = cv2.resize(img, (512, 512))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        i += 1
        print(i, '/', len_)

    batch_imgs = np.concatenate(img_list, axis=3)
    batch_imgs = batch_imgs.astype(np.float32) / 255.

    for k in range(3):
        pixels = batch_imgs[:, :, k, :].ravel()
        batch_mean.append(np.mean(pixels))
        batch_std.append(np.std(pixels))
        print("BatchMean = {}".format(batch_mean))
        print("BatchStd = {}".format(batch_std))

    mean += np.array(batch_mean)
    std += np.array(batch_std)
    print("SumMean = {}".format(mean))
    print("SumStd = {}".format(std))
    img_list.clear()
    print("--------", j + 1, '/', step)

# std为每个batch的std的均值，属于有偏估计
final_mean = np.array(mean) / step
final_std = np.array(std) / step
final_mean = list(final_mean[::-1])
final_std = list(final_std[::-1])

print("RGBMean = {}".format(final_mean))
print("RGBStd = {}".format(final_std))
