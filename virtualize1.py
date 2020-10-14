import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
import json
from networks_m import *

# 图片预处理
def img_preprocess(img_in):
    img = img_in.copy()
    img = img[:, :, ::-1]   				# 1
    img = np.ascontiguousarray(img)			# 2
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
    ])
    img = transform(img)
    img = img.unsqueeze(0)					# 3
    return img

def normalize(data):
    return data / 255.

# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)

# 计算grad-cam并可视化
def cam_show_img(img, feature_map, grads, out_dir):
    H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		# 4
    grads = grads.reshape([grads.shape[0],-1])					# 5
    weights = np.mean(grads, axis=1)							# 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]							# 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = 0.3 * heatmap + 0.7 * img

    path_cam_img = os.path.join(out_dir, "cam.jpg")
    cv2.imwrite(path_cam_img, cam_img)


if __name__ == '__main__':
    path_img = './noclean278.png'
    json_path = './cam/labels.json'
    output_dir = './logs'

    # with open(json_path, 'r') as load_f:
    #     load_json = json.load(load_f)
    # classes = {int(key): value for (key, value)
    #            in load_json.items()}

    # 只取标签名
    # classes = list(classes.get(key) for key in range(1000))

    # 存放梯度和特征图
    fmap_block = list()
    grad_block = list()

    # 图片读取；网络加载
    img = cv2.imread(path_img, 1)
    # img_input = img_preprocess(img)

    b, g, r = cv2.split(img)
    y = cv2.merge([r, g, b])
    # y = cv2.resize(y, (int(500), int(500)), interpolation=cv2.INTER_CUBIC)
    y = normalize(np.float32(y))
    y = np.expand_dims(y.transpose(2, 0, 1), 0)
    img_input = Variable(torch.Tensor(y))

    # 加载 squeezenet1_1 预训练模型
    # net = models.squeezenet1_1(pretrained=False)
    net = PReNet(recurrent_iter=2, use_GPU=False)
    # pthfile = './net_latest.pth'
    # net.load_state_dict(torch.load(pthfile))
    net.load_state_dict(torch.load( 'net_latest.pth', map_location='cpu'))
    net.eval()  # 8
    print(net)
    # out, _ = model(y)
    # out = torch.clamp(out, 0., 1.)
    # 注册hook
    # net.features[-1].expand3x3.register_forward_hook(farward_hook)  # 9
    # net.features[-1].expand3x3.register_backward_hook(backward_hook)
    net.conv[-1].register_forward_hook(farward_hook)
    net.conv[-1].register_forward_hook(backward_hook)

    # net.self_attn.key_conv.register_forward_hook(farward_hook)
    # net.self_attn.key_conv.register_forward_hook(backward_hook)
    # net.res_conv5[-1].register_forward_hook(farward_hook)
    # net.res_conv5[-1].register_forward_hook(backward_hook)

    # net.res_conv2.
    # net.self_attn.softmax.re
    # forward
    output,_ = net(img_input)
    output = torch.clamp(output, 0., 1.)
    idx = np.argmax(output.cpu().data.numpy())
    # print("predict: {}".format(classes[idx]))

    # backward
    # net.zero_grad()
    # class_loss = output[0]
    # class_loss.backward()

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()

    # 保存cam图片
    cam_show_img(img, fmap, grads_val, output_dir)

    save_out = np.uint8(255 * output.data.numpy().squeeze())
    save_out = save_out.transpose(1, 2, 0)
    b, g, r = cv2.split(save_out)
    save_out = cv2.merge([r, g, b])

    cv2.imwrite('modelOut.png', save_out)
