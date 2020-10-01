import cv2
import os

for i in range(50):
    prefix1 = "noclean-"
    data_path1 = "E:\\lzb\\removePic626\\dirty"
    prefix2 = "clean-"
    data_path2 = "E:\\lzb\\removePic626\\remove"

    data_path = os.path.join(data_path1)
    target_path = "datasets/train/BladeTrainL"
    target_file = "%03d.png" % (i + 1)

    # 含附着物
    # print(os.path.join(data_path1, target_file))
    target = cv2.imread(os.path.join(data_path1, target_file))
    print(os.path.join(target_path, prefix1 + target_file))
    cv2.imwrite(os.path.join(target_path, prefix1 + target_file),target)

    # 不含附着物
    # print(os.path.join(data_path2, target_file))
    # target = cv2.imread(os.path.join(data_path2, target_file))
    # print(os.path.join(target_path, prefix2 + target_file))
    # cv2.imwrite(os.path.join(target_path, prefix2 + target_file),target)

    # cv2.imshow("test",target)
    # cv2.waitKey(0);