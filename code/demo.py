# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import glob, cv2, torch
import numpy as np
from os.path import realpath, dirname, join

from net import SiamRPNvot
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect

# load net
net = SiamRPNvot()
net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNVOT.model')))
net.eval().cuda()

# image and init box

#image_files = sorted(glob.glob('./human5/*.jpg'))
#init_rbox = [326.0, 419.0, 338.0, 420.0, 336.0, 453.0, 326.0, 453.0] # human5
#image_files = sorted(glob.glob('./bag/*.jpg'))
#init_rbox = [334.02,128.36,438.19,188.78,396.39,260.83,292.23,200.41] # bag
image_files = sorted(glob.glob('./rajeev/*.jpg'))
init_rbox = [16.0, 445.0, 158.0, 445.0, 158.0, 715.0, 16.0, 715.0] # rajeev
[cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)

# tracker init
target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
im = cv2.imread(image_files[0])  # HxWxC
state = SiamRPN_init(im, target_pos, target_sz, net)

# tracking and visualization
toc = 0
for f, image_file in enumerate(image_files):
    im = cv2.imread(image_file)
    tic = cv2.getTickCount()
    state = SiamRPN_track(state, im)  # track
    toc += cv2.getTickCount()-tic
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
    res = [int(l) for l in res]
    cv2.rectangle(im, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
    cv2.imshow('SiamRPN', im)
    cv2.waitKey(1)

print('Tracking Speed {:.1f}fps'.format((len(image_files)-1)/(toc/cv2.getTickFrequency())))
