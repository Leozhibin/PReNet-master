import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
from utils import *
from networks import *
import time 



def main():

    #model = PReNet_x(6, False)
    #model = PRN_r(6, False)
    #model = PRN(6, False)
    # model = PReNet(6, False)
    # model = PReNet_x(6, False)
    #model = PReNet_r(6, False)
    #model = PReNet_LSTM(6, False)
    model = PReNet_GRU(6, False)

    print_network(model)
    print("hi")

main()
