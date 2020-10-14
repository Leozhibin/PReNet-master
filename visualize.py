import sys
sys.path.append('/')

import cv2
import numpy as np
from torchvision import models
# from efficientnet_pytorch import EfficientNet
from grad_cam import GradCam,GuidedBackpropReLUModel,show_cams,show_gbs,preprocess_image
from networks_m import *


# model = models.vgg19(pretrained=True)
# print(model)
model = PReNet(recurrent_iter=2, use_GPU=False)
model.load_state_dict(torch.load('net_latest.pth'))
model.eval()
grad_cam = GradCam(model=model, blob_name = 'conv', target_layer_names=['0'], use_cuda=False)
# grad_cam = GradCam(model=model, blob_name = 'self_attn', target_layer_names=['query_conv','key_conv','value_conv'], use_cuda=False)
# grad_cam = GradCam(model=model, blob_name = 'features', target_layer_names=['4','20','36'], use_cuda=False)


img = cv2.imread('./noclean530.png')
# img = cv2.imread('./dog.jpg', 1)
# img = np.float32(img) / 255
img = np.float32(cv2.resize(img, (100, 100))) / 255

inputs = preprocess_image(img)
# If None, returns the map for the highest scoring category.
# Otherwise, targets the requested index.
target_index = None
mask_dic = grad_cam(inputs, target_index)
show_cams(img, mask_dic)
gb_model = GuidedBackpropReLUModel(model=model, activation_layer_name = 'ReLU', use_cuda=False)
show_gbs(inputs, gb_model, target_index, mask_dic)