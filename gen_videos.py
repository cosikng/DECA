import cv2
import os
from tqdm import tqdm

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 设置输出视频为mp4格式
# cap_fps是帧率，根据自己需求设置帧率
cap_fps = 30

# size要和图片的size一样，但是通过img.shape得到图像的参数是（height，width，channel），
# 可以实现在图片文件夹下查看图片属性，获得图片的分辨率
size = (224, 224)  # size（width，height）
# 设置输出视频的参数，如果是灰度图，可以加上 isColor = 0 这个参数
# video = cv2.VideoWriter('results/result.avi',fourcc, cap_fps, size, isColor=0)
video = cv2.VideoWriter('result.mp4', fourcc, cap_fps, size)  # 设置保存视频的名称和路径，默认在根目录下

path = 'TestSamples/my_exps/'  # 设置图片文件夹的路径，末尾加/
file_lst = os.listdir(path)
for filename in tqdm(file_lst):
    img = cv2.resize(cv2.imread(path + filename), (224, 224))
    video.write(img)
video.release()
