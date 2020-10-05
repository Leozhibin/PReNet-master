import cv2
import numpy as np
import matplotlib.pyplot as plt
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

output = cv2.imread("logs/noclean125.png")
input = cv2.imread("logs/noclean125test.png")
diff = diffimage(output,input)
cv2.imwrite("logs/diff.png",diff)
# cv2.imshow("diff",diff*255)
img = diff

bins = np.arange(257)
item = img[:, :, 1]
hist, bins = np.histogram(item, bins)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()

