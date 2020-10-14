import numpy as np
import cv2
import torch
import os
def diffimage(src_1, src_2):
    # 将numpy的基础类型转化为int型否则uint8会发生溢出
    src_1 = src_1.astype(np.int)
    src_2 = src_2.astype(np.int)
    diff = abs(src_1 - src_2)
    return diff.astype(np.uint8)

def calcAndDrawHist(image, color):
    hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256, 256, 3], np.uint8)
    hpt = int(0.9 * 256);

    for h in range(256):
        intensity = int(hist[h] * hpt / maxVal)
        cv2.line(histImg, (h, 256), (h, 256 - intensity), color)

    return histImg;

def get_metrics(target, predict):
    tp = (predict * target).sum()
    tn = ((1 - predict) * (1 - target)).sum()
    fp = ((1 - target) * predict).sum()
    fn = ((1 - predict) * target).sum()
    acc = (tp + tn) / (tp + fp + fn + tn)
    pre = tp / (tp + fp)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    iou = tp / (tp + fp + fn)
    f1 = 2 * pre * sen / (pre + sen)
    return {
        # "Acc": np.round(acc, 4),
        # "pre": np.round(pre, 4),
        # "Sem": np.round(sen, 4),
        # "Spe": np.round(spe, 4),
        # "F1": np.round(f1, 4),
        "IOU": np.round(iou, 4)
    }

rootdir = 'C:\\Users\\DELL\\Desktop\\fsdownload\\PReNet_Atte_channel32To16'
rootdir1 = 'C:\\Users\\DELL\\Desktop\\fsdownload\\patch\\'

list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
allIOU=0.0
for i in range(0,len(list)):
    # print(len(list))
    # print(i)
    modelOutPath = os.path.join(rootdir,list[i])
    gtPath = os.path.join(rootdir1+'clean567',list[i][2:])
    rawPath = os.path.join(rootdir1+'noclean567',list[i])
    # if os.path.isfile(path):

    raw = cv2.imread(rawPath)
    modelOut = cv2.imread(modelOutPath)
    gt = cv2.imread(gtPath)
    # print(rawPath)

    # if list[i]=="noclean440.png":
    #     print(modelOutPath)
    #     print(gtPath)
    #     print(rawPath)
    #     cv2.imshow("gt", gt)
    #     cv2.imshow("modelOut", modelOut)
    #     cv2.imshow("raw", raw)
    #     # cv2.imshow("diff", diffimage(raw,modelOut))
    #     # cv2.imshow("diff1", diffimage(raw,gt))
    #     cv2.waitKey(0)
    diff1 = diffimage(raw, modelOut)/255
    diff2 = diffimage(raw,gt)/255
    inference = torch.tensor(diff1)
    gt = torch.tensor(diff2)
    inference_b = torch.where(inference >= 0.1, torch.full_like(inference, 1), torch.full_like(inference, 0))
    gt_b = torch.where(gt >= 0.1, torch.full_like(gt, 1), torch.full_like(gt, 0))
    print(get_metrics(inference_b,gt_b))
    allIOU = allIOU+ get_metrics(inference_b,gt_b).get('IOU')
print(allIOU/len(list))


