import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import *
import random

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)

    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def Im2PatchRaw100(img,img1, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    patch1 = img1[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    # nonZeroNum=0
    # for i in range(win):
    #     for j in range(win):
    #         patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
    #         patch1 = img1[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
    #         patchSub = patch1 - patch
    #         sumPixel = np.sum(patchSub)
    #         # print("sum " + str(sumPixel))
    #         if(sumPixel!=0.0):
    #             nonZeroNum+=1

    # TotalPatNum = nonZeroNum
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)
    Y1 = np.zeros([endc, win * win, TotalPatNum], np.float32)

    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            patch1 = img1[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            patchSub = patch1 - patch
            sumPixel = np.sum(patchSub)
            # print("sum " + str(sumPixel))
            Y1[:, k, :] = np.array(patch1[:]).reshape(endc, TotalPatNum)
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1

    return Y.reshape([endc, win, win, TotalPatNum]),Y1.reshape([endc, win, win, TotalPatNum])


def Im2Patch2(img,img1, win, stride=1):
    subImg1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    subImg2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    subImg = cv2.subtract(subImg1, subImg2)

    ret, binary = cv2.threshold(subImg, 20, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for t1 in contours:
        leftX=[]
        leftY=[]
        rightX = []
        rightY =[]
        leftx=5000
        lefty=0
        rightx=-1
        righty=0
        for t2 in t1:
            for t3 in t2:
                if t3[0]<leftx:
                    leftx=t3[0]
                    lefty=t3[1]
                if t3[0]>rightx:
                    rightx=t3[0]
                    righty=t3[1]
        leftX.append(leftx)
        leftY.append(lefty)
        rightX.append(rightx)
        rightY.append(righty)
        # print(str(leftx)+ " "+str(lefty))
        # print(str(rightx)+ " "+str(righty))
        # print(" ")

    # M = cv2.moments(contours[0])  # 计算第一条轮廓的各阶矩,字典形式
    # center_x.append(int(M["m10"] / M["m00"]))
    # center_y.append(int(M["m01"] / M["m00"]))


    img = np.float32(normalize(img))
    img = img.transpose(2, 0, 1)
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    # print(endw - win + 0 + 1)
    # print(endh - win + 0 + 1)
    # patch = img[:, :, win]
    patchCount = len(contours)*len(leftX)*4
    # patch = img[:, :, patchCount]

    # TotalPatNum = patch.shape[1] * patch.shape[2]
    # Y = np.zeros([endc, win, win,patchCount], np.float32)
    Y=[]
    # 3 700 4096
    # 3 700 43
    # 0 3397 80
    print("len(contours) " +str(len(contours)))
    if len(contours)>20:
        numPerImg = 1
    else:
        numPerImg = 4

    for i in range(len(contours)):
        for j in range(len(leftX)):
            for g in range(numPerImg):
                moveTotal = win - (rightX[j] - leftX[j])
                movex = random.randint(0, moveTotal)
                moveRemain = moveTotal-movex
                cutleft = leftX[j] - movex
                cutright =rightX[j] + moveRemain
                if((cutleft>0)and(cutright<= endh)):
                    patch = img[:, :, cutleft:cutright]
                    # np.concatenate((a, b), axis=0)
                    Y.append(patch)
                    # Y[:, :, :,k] = np.array(patch[:]).reshape(endc, patchCount)
                    k = k + 1
    YNnmpy = np.array(Y)
    # print(YNnmpy.shape)
    YNnmpyTran = YNnmpy.transpose(1,2,3,0)
    return YNnmpyTran

def Im2Patch1(img,img1, win, stride=1):
    # img.transpose(1, 0, 2)
    # img1.transpose(1, 0, 2)
    subImg1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    subImg2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    subImg = cv2.subtract(subImg2, subImg1)
    # # cv2.namedWindow("Lu1", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("Lu2", cv2.WINDOW_NORMAL)
    # # cv2.imshow("Lu1", subImg1)
    # cv2.imshow("Lu2", subImg)
    # cv2.waitKey(0)
    center_x=[]
    center_y=[]
    ret, binary = cv2.threshold(subImg, 20, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for t1 in contours:
        leftX=[]
        leftY=[]
        rightX = []
        rightY =[]
        leftx=5000
        lefty=0
        rightx=-1
        righty=0
        for t2 in t1:
            for t3 in t2:
                if t3[0]<leftx:
                    leftx=t3[0]
                    lefty=t3[1]
                if t3[0]>rightx:
                    rightx=t3[0]
                    righty=t3[1]
        leftX.append(leftx)
        leftY.append(lefty)
        rightX.append(rightx)
        rightY.append(righty)
        # print(str(leftx)+ " "+str(lefty))
        # print(str(rightx)+ " "+str(righty))
        # print(" ")

    # M = cv2.moments(contours[0])  # 计算第一条轮廓的各阶矩,字典形式
    # center_x.append(int(M["m10"] / M["m00"]))
    # center_y.append(int(M["m01"] / M["m00"]))


    img = np.float32(normalize(img))
    img = img.transpose(2, 0, 1)
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    # print(endw - win + 0 + 1)
    # print(endh - win + 0 + 1)
    # patch = img[:, :, win]
    patchCount = len(contours)*len(leftX)*4
    # patch = img[:, :, patchCount]

    # TotalPatNum = patch.shape[1] * patch.shape[2]
    # Y = np.zeros([endc, win, win,patchCount], np.float32)
    Y=[]
    # 3 700 4096
    # 3 700 43
    # 0 3397 80
    if len(contours)>20:
        numPerImg = 1
    else:
        numPerImg = 4
    for i in range(len(contours)):
        for j in range(len(leftX)):
            for g in range(numPerImg):
                moveTotal = win - (rightX[j] - leftX[j])

                movex = random.randint(0, moveTotal)
                moveRemain = moveTotal-movex
                cutleft = leftX[j] - movex
                cutright =rightX[j] + moveRemain
                if((cutleft>0)and(cutright<= endh)):
                    patch = img[:, :, cutleft:cutright]
                    # np.concatenate((a, b), axis=0)
                    Y.append(patch)
                    # Y[:, :, :,k] = np.array(patch[:]).reshape(endc, patchCount)
                    k = k + 1
    YNnmpy = np.array(Y)
    # print("YNnmpy.shape")
    # print(YNnmpy.shape)
    YNnmpyTran = YNnmpy.transpose(1,2,3,0)
    return YNnmpyTran

def Im2PatchTwoImage(img,img1, win, stride=1):
    # img.transpose(1, 0, 2)
    # img1.transpose(1, 0, 2)
    subImg1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    subImg2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    subImg = cv2.subtract(subImg2, subImg1)
    # # cv2.namedWindow("Lu1", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("Lu2", cv2.WINDOW_NORMAL)
    # # cv2.imshow("Lu1", subImg1)
    # cv2.imshow("Lu2", subImg)
    # cv2.waitKey(0)
    center_x=[]
    center_y=[]
    ret, binary = cv2.threshold(subImg, 20, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    leftX = []
    leftY = []
    rightX = []
    rightY = []
    for t1 in contours:

        leftx=5000
        lefty=0
        rightx=-1
        righty=0
        for t2 in t1:
            for t3 in t2:
                if t3[0]<leftx:
                    leftx=t3[0]
                    lefty=t3[1]
                if t3[0]>rightx:
                    rightx=t3[0]
                    righty=t3[1]
        leftX.append(leftx)
        leftY.append(lefty)
        rightX.append(rightx)
        rightY.append(righty)
        # print(str(leftx)+ " "+str(lefty))
        # print(str(rightx)+ " "+str(righty))
        # print(" ")

    # M = cv2.moments(contours[0])  # 计算第一条轮廓的各阶矩,字典形式
    # center_x.append(int(M["m10"] / M["m00"]))
    # center_y.append(int(M["m01"] / M["m00"]))


    img = np.float32(normalize(img))
    img1 = np.float32(normalize(img1))
    img = img.transpose(2, 0, 1)
    img1 = img1.transpose(2, 0, 1)
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    # print(endw - win + 0 + 1)
    # print(endh - win + 0 + 1)
    # patch = img[:, :, win]
    patchCount = len(contours)*len(leftX)*4
    # patch = img[:, :, patchCount]

    # TotalPatNum = patch.shape[1] * patch.shape[2]
    # Y = np.zeros([endc, win, win,patchCount], np.float32)
    Y=[]
    Y1=[]
    # 3 700 4096
    # 3 700 43
    # 0 3397 80
    if len(leftX)>20:
        numPerImg = 1
    else:
        numPerImg = 2
    for i in range(len(leftX)):
        # for j in range(len(leftX)):
        # print("len(leftX)")
        # print(len(leftX))
        indexNow =0
        for g in range(numPerImg):
            moveTotal = win - (rightX[i] - leftX[i])

            movex = random.randint(0, moveTotal)
            moveRemain = moveTotal-movex
            cutleft = leftX[i] - movex
            cutright =rightX[i] + moveRemain
            if((cutleft>=0)and(cutright<= endh)):
                indexNow= indexNow+1
                patch = img[:, :, cutleft:cutright]
                patch1 = img1[:, :, cutleft:cutright]
                # np.concatenate((a, b), axis=0)
                Y.append(patch)
                Y1.append(patch1)
                # Y[:, :, :,k] = np.array(patch[:]).reshape(endc, patchCount)
                k = k + 1
            else:
                continue

    YNnmpy = np.array(Y)
    YNnmpy1 = np.array(Y1)
    if indexNow==0:
        return YNnmpy,YNnmpy1
    # print("=====")
    # print("YNnmpy.shape")
    # print(YNnmpy.shape)
    # print("YNnmpy.shape")
    # print(YNnmpy.shape)
    # print("indexNow")
    # print(indexNow)

    YNnmpyTran = YNnmpy.transpose(1,2,3,0)
    YNnmpyTran1 = YNnmpy1.transpose(1,2,3,0)
    return YNnmpyTran,YNnmpyTran1


def prepare_data_Rain12600(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path, 'rainy_image')
    target_path = os.path.join(data_path, 'ground_truth')

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(900):
        target_file = "%d.jpg" % (i + 1)
        target = cv2.imread(os.path.join(target_path,target_file))
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        for j in range(14):
            input_file = "%d_%d.jpg" % (i+1, j+1)
            input_img = cv2.imread(os.path.join(input_path,input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])

            target_img = target
            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)

            input_img = np.float32(normalize(input_img))
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)
            print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))

            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)
                train_num += 1

    target_h5f.close()
    input_h5f.close()
    print('training set, # samples %d\n' % train_num)


def prepare_data_RainTrainH(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(1800):
        target_file = "norain-%d.png" % (i + 1)
        if os.path.exists(os.path.join(target_path,target_file)):

            target = cv2.imread(os.path.join(target_path,target_file))
            b, g, r = cv2.split(target)
            target = cv2.merge([r, g, b])

            input_file = "rain-%d.png" % (i + 1)

            if os.path.exists(os.path.join(input_path,input_file)): # we delete 546 samples

                input_img = cv2.imread(os.path.join(input_path,input_file))
                b, g, r = cv2.split(input_img)
                input_img = cv2.merge([r, g, b])

                target_img = target
                target_img = np.float32(normalize(target_img))
                target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)

                input_img = np.float32(normalize(input_img))
                input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

                print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))

                for n in range(target_patches.shape[3]):
                    target_data = target_patches[:, :, :, n].copy()
                    target_h5f.create_dataset(str(train_num), data=target_data,shape=(3,700,700))

                    input_data = input_patches[:, :, :, n].copy()
                    input_h5f.create_dataset(str(train_num), data=input_data,shape=(3,700,700))

                    train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)

def prepare_data_RainTrainH1(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(100):
        target_file = "norain-%03d.png" % (i + 1)
        if os.path.exists(os.path.join(target_path,target_file)):

            target = cv2.imread(os.path.join(target_path,target_file))
            b, g, r = cv2.split(target)
            target = cv2.merge([r, g, b])

            input_file = "rain-%03d.png" % (i + 1)

            if os.path.exists(os.path.join(input_path,input_file)): # we delete 546 samples

                input_img = cv2.imread(os.path.join(input_path,input_file))
                b, g, r = cv2.split(input_img)
                input_img = cv2.merge([r, g, b])

                target_img = target
                target_img = np.float32(normalize(target_img))
                target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)

                input_img = np.float32(normalize(input_img))
                input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

                print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))

                for n in range(target_patches.shape[3]):
                    target_data = target_patches[:, :, :, n].copy()
                    target_h5f.create_dataset(str(train_num), data=target_data)

                    input_data = input_patches[:, :, :, n].copy()
                    input_h5f.create_dataset(str(train_num), data=input_data)

                    train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)

def prepare_data_BladeTrainL(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(50):
        # i = 2
        target_file = "clean-%03d.png" % (i + 1)
        target = cv2.imread(os.path.join(target_path,target_file))
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        for j in range(2):
            input_file = "noclean-%03d.png" % (i + 1)
            input_img = cv2.imread(os.path.join(input_path,input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])
            target_img = target


            # if j == 1:
            #     target_img = cv2.flip(target_img, 1)
            #     input_img = cv2.flip(input_img, 1)

            # input_img = np.float32(normalize(input_img))
            # target_img = np.float32(normalize(target_img))

            input_patches,target_patches = Im2PatchTwoImage(input_img,target_img, win=patch_size, stride=stride)
            if(len(input_patches) ==0and len(target_patches)==0):
                continue
            # target_patches = Im2Patch1(input_img,target_img, win=patch_size, stride=stride)
            # input_patches = Im2Patch2(target_img,input_img, win=patch_size, stride=stride)

            # print("input_patches " + str(input_patches.shape))
            # print("target_patches " + str(target_patches.shape))
            print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))
            for n in range(target_patches.shape[3]):
                # print("++++++ n " + str(n))
                # print("target_patches.shape")
                # print(target_patches.shape)
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                # print("input_patches.shape")
                # print(input_patches.shape)
                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)

                # show
                # b1 = input_data[0, :, :]
                # g1 = input_data[1, :, :]
                # r1 = input_data[2, :, :]
                # b2 = target_data[0, :, :]
                # g2 = target_data[1, :, :]
                # r2 = target_data[2, :, :]
                # imgShow1 = cv2.merge([b1, g1, r1])
                # imgShow2 = cv2.merge([b2, g2, r2])
                # cv2.namedWindow("Lu1", cv2.WINDOW_NORMAL)
                # cv2.namedWindow("Lu2", cv2.WINDOW_NORMAL)
                # cv2.imshow("Lu1", imgShow1)
                # cv2.imshow("Lu2", imgShow2)
                # cv2.waitKey(0)

                train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)

def prepare_data_RainTrainL(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    # for i in range(50):#测试集
    for i in range(10): #训练集
        # target_file = "clean-%03d.png" % (i + 1) #训练集
        target_file = "clean-%03d.png" % (i + 51) #测试集
        target = cv2.imread(os.path.join(target_path,target_file))
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        for j in range(2):
            # input_file = "noclean-%03d.png" % (i + 1) #训练集
            input_file = "noclean-%03d.png" % (i + 51) #测试集
            input_img = cv2.imread(os.path.join(input_path,input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])

            target_img = target

            if j == 1:
                target_img = cv2.flip(target_img, 1)
                input_img = cv2.flip(input_img, 1)

            # target_img = np.float32(normalize(target_img))
            # input_img = np.float32(normalize(input_img))
            # input_patches,target_patches = Im2PatchRaw100(input_img.transpose(2, 0, 1),target_img.transpose(2,0,1), win=patch_size, stride=stride)

            target_img = np.float32(normalize(target_img))
            input_img = np.float32(normalize(input_img))
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)
            target_patches = Im2Patch(target_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                input_data = input_patches[:, :, :, n].copy()

                patchSub = input_data - target_data
                sumPixel = np.sum(patchSub)
                # 让附着物居中在图片中央
                nonzeroxy = patchSub[0, :, :].nonzero()
                nonzerox = np.mean(nonzeroxy[0])
                nonzeroy = np.mean(nonzeroxy[1])

                if(sumPixel!=0.0 and nonzerox>50.0 and nonzerox<150.0 and nonzeroy>50.0 and nonzeroy<150.0):
                    # 将200x200 resize成100x100
                    # target_data= target_data.transpose(1, 2, 0)
                    # target_data = cv2.resize(target_data, dsize=(100, 100), interpolation=cv2.INTER_NEAREST)
                    # target_data = target_data.transpose(2, 0, 1)

                    # input_data= input_data.transpose(1, 2, 0)
                    # input_data = cv2.resize(input_data, dsize=(100, 100), interpolation=cv2.INTER_NEAREST)
                    # input_data = input_data.transpose(2, 0, 1)

                    target_h5f.create_dataset(str(train_num), data=target_data)
                    input_h5f.create_dataset(str(train_num), data=input_data)
                    train_num += 1
            # print("sum " + str(train_num))
            print("target file: %s # samples: %d" % (input_file, train_num))
    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)


class Dataset(udata.Dataset):
    def __init__(self, data_path='.'):
        super(Dataset, self).__init__()

        self.data_path = data_path

        target_path = os.path.join(self.data_path, 'train_target.h5')
        input_path = os.path.join(self.data_path, 'train_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        target_path = os.path.join(self.data_path, 'train_target.h5')
        input_path = os.path.join(self.data_path, 'train_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key])
        input = np.array(input_h5f[key])

        target_h5f.close()
        input_h5f.close()

        return torch.Tensor(input), torch.Tensor(target)


