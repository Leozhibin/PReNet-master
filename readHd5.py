import h5py
import numpy as np
import cv2
import os

from PIL import Image

# target_path = "E:\\lzb\\derain\\PReNet-master\\datasets\\train\\BladeTrainL"
target_path = "D:\\code\\PReNet-master-master\\datasets\\test\\BladeTestL"
print(os.path.join(target_path+"train_input.h5"))
train_inputs = h5py.File(os.path.join(target_path,"train_input.h5"),'r')#//此文件下的data数据
train_targets = h5py.File(os.path.join(target_path,"train_target.h5"),'r')#//此文件下的data数据
# train_targets = h5py.File(target_path+"/train_target.h5")['data']#//此文件下的lable数据
lenght=len(train_inputs)#//获取数据的长度
for i in range(len(train_inputs)):#//for循环提取数据
    input_data = train_inputs[str(i)][:]
    target_data = train_targets[str(i)][:]

    print()
    # show
    b1 = input_data[0, :, :]
    g1 = input_data[1, :, :]
    r1 = input_data[2, :, :]
    b2 = target_data[0, :, :]
    g2 = target_data[1, :, :]
    r2 = target_data[2, :, :]
    noclean = cv2.merge([b1, g1, r1])
    clean = cv2.merge([b2, g2, r2])
    cv2.namedWindow("Lu1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Lu2", cv2.WINDOW_NORMAL)
    cv2.imshow("Lu1", noclean)
    cv2.imshow("Lu2", clean)
    cv2.waitKey(0)

    #保存test文件 patch100
    # print("datasets\\test\BladeTestL\patch\noclean" + str(i)+".png")
    # print(cv2.imwrite("datasets\\test\BladeTestL\patch\\noclean" + str(i)+".png",noclean*255))
    # print(cv2.imwrite("datasets\\test\BladeTestL\patch\clean" + str(i)+".png",clean*255))
