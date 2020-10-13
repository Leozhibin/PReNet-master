import cv2
import numpy as np
import copy
import os
import shapely
from shapely.geometry import Polygon,MultiPoint

def diffimage(src_1, src_2):
    # 将numpy的基础类型转化为int型否则uint8会发生溢出
    src_1 = src_1.astype(np.int)
    src_2 = src_2.astype(np.int)
    diff = abs(src_1 - src_2)
    return diff.astype(np.uint8)

def getBox(image):
    B, G, R = cv2.split(image)
    ret, thresh = cv2.threshold(G, 50, 255, cv2.THRESH_BINARY)
    # print(thresh.shape)
    # 单通道复制为三通道  ...代替所有的：：   3 增加后的通道数   2 轴
    GGG = np.repeat(G[...,np.newaxis], 3, 2)
    # print(GGG.shape)
    eye = np.bitwise_and(image, GGG)
    # 1. 找到所有的点, 画点的最小外接矩形
    coords = np.column_stack(np.where(thresh > 0))
    coords = coords[:, ::-1] # x, y互换
    min_rect = cv2.minAreaRect(coords)
    box = cv2.boxPoints(min_rect)
    box = np.int0(box)
    eye1 = copy.deepcopy(eye)
    box1 = cv2.drawContours(eye1, [box], 0, [0, 255, 0], 1)

    # minx=1000000
    # miny=1000000
    # maxx=0
    # maxy=0
    # boxminmax=[]
    # for x,y in box:
    #     minx = min(x,minx)
    #     maxx = max(x,maxx)
    #     miny = min(y,miny)
    #     maxy = max(y,maxy)
    # # print(str(minx) + str(miny) + str(maxx) + str(maxy))
    # boxminmax.append(minx)
    # boxminmax.append(miny)
    # boxminmax.append(maxx)
    # boxminmax.append(maxy)

    # cv2.imshow('eye1', box1)
    # cv2.waitKey(0)
    return box

def get_iou(a, b):
    '''
    :param a: box a [x0,y0,x1,y1,x2,y2,x3,y3]
    :param b: box b [x0,y0,x1,y1,x2,y2,x3,y3]
    :return: iou of bbox a and bbox b
    '''
    a = a.reshape(4, 2)
    poly1 = Polygon(a).convex_hull

    b = b.reshape(4, 2)
    poly2 = Polygon(b).convex_hull

    if not poly1.intersects(poly2):  # 如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            union_area = poly1.area + poly2.area - inter_area
            if union_area == 0:
                iou = 0
            else:
                iou = inter_area / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou


def compute_iou(box1, box2, wh=False):
    """
    compute the iou of two boxes.
    Args:
        box1, box2: [xmin, ymin, xmax, ymax] (wh=False) or [xcenter, ycenter, w, h] (wh=True)
        wh: the format of coordinate.
    Return:
        iou: iou of box1 and box2.
    """
    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0] - box1[2] / 2.0), int(box1[1] - box1[3] / 2.0)
        xmax1, ymax1 = int(box1[0] + box1[2] / 2.0), int(box1[1] + box1[3] / 2.0)
        xmin2, ymin2 = int(box2[0] - box2[2] / 2.0), int(box2[1] - box2[3] / 2.0)
        xmax2, ymax2 = int(box2[0] + box2[2] / 2.0), int(box2[1] + box2[3] / 2.0)

    ## 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    ## 计算两个矩形框面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)

    return iou

rootdir = 'C:\\Users\\Leonard\\Desktop\\fsdownload\\resultpatch200MobileExpand1No1x1Atte100_r2_rightv2'
rootdir1 = 'C:\\Users\\Leonard\\Desktop\\fsdownload\\patch\\'

list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
allIOU=0.0
for i in range(0,len(list)):
    # print(len(list))
    # print(i)
    modelOutPath = os.path.join(rootdir,list[i])
    gtPath = os.path.join(rootdir1+'clean567',list[i][2:])
    rawPath = os.path.join(rootdir1+'noclean567',list[i])
    # if os.path.isfile(path):

    dirty = cv2.imread(rawPath)
    output = cv2.imread(modelOutPath)
    gt = cv2.imread(gtPath)


    imageGT = diffimage(gt,dirty)
    imageOUT = diffimage(output,dirty)

    # print(calBox(imageGT))
    # print(calBox(imageOUT))
    i = get_iou(getBox(imageGT), getBox(imageOUT))
    print(i)
    allIOU = allIOU+i
print(allIOU/len(list))
# print(boxminmax)
# 2. 找轮廓, 包含所有轮廓的点
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contour = []
# for cont in contours:
#     contour.extend(cont)
#
# min_rect = cv2.minAreaRect(np.array(contour))
# box = cv2.boxPoints(min_rect)
# box = np.int0(box)
# eye2 = copy.deepcopy(eye)
# box2 = cv2.drawContours(eye2, [box], 0, [0, 0, 255], 1)

# cv2.imshow('eye1', box1)
# cv2.imshow('eye2', box2)
# cv2.waitKey(0)

