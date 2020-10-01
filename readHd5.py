import h5py
import numpy as np
import cv2
import os

from PIL import Image

# target_path = "E:\\lzb\\derain\\PReNet-master\\datasets\\train\\RainTrainL"
target_path = "./datasets/test/BladeTestL"
print(os.path.join(target_path+"train_input.h5"))
train_inputs = h5py.File(os.path.join(target_path,"train_input.h5"),'r')#//此文件下的data数据
# train_targets = h5py.File(os.path.join(target_path,"train_target.h5"),'r')#//此文件下的data数据
# train_targets = h5py.File(target_path+"/train_target.h5")['data']#//此文件下的lable数据
lenght=len(train_inputs)#//获取数据的长度
for i in range(len(train_inputs)):#//for循环提取数据
    input_data = train_inputs[str(i)][:]
    # target_data = train_targets[str(i)][:]

    print()
    # show
    b1 = input_data[0, :, :]
    g1 = input_data[1, :, :]
    r1 = input_data[2, :, :]
    # b2 = target_data[0, :, :]
    # g2 = target_data[1, :, :]
    # r2 = target_data[2, :, :]
    imgShow1 = cv2.merge([b1, g1, r1])
    # imgShow2 = cv2.merge([b2, g2, r2])
    #cv2.namedWindow("Lu1", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("Lu2", cv2.WINDOW_NORMAL)
    #cv2.imshow("Lu1", imgShow1)
    # cv2.imshow("Lu2", imgShow2)
    #cv2.waitKey(0)

    # cv2.imwrite(str(i)+"1.png",y)
    # cv2.imwrite(str(i)+"0.png",x)

    
    target_file = "%04d.png" % (i )
    print(os.path.join(target_path+target_file))
    cv2.imwrite(os.path.join(target_path+target_file),imgShow1*255)
