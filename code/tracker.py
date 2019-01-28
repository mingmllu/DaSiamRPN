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

import time
import os


# load net
net = SiamRPNvot()
net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNVOT.model')))
net.eval().cuda()

image_source = 'demo.mp4'
if os.getenv('IMAGE_SOURCE'):
  image_source = os.getenv('IMAGE_SOURCE')

def open_source_video(image_source):
  # allow multiple attempts to open video source
  max_num_attempts = 10
  count_attempts = 1
  cap = cv2.VideoCapture(image_source)
  # Check if camera opened successfully
  while (cap.isOpened() == False):
    print("Unable to open image source %s: %d out of %d"%(image_source, count_attempts, max_num_attempts))
    if count_attempts == max_num_attempts:
      break
    time.sleep(0.5)
    count_attempts += 1
    cap = cv2.VideoCapture(image_source)
  return cap # return a video capture object that is in open state 


def init_SiamRPN(cap, init_rbox, skipped_frames = 1):
  # Use the first valid image to initialize the DaSiamRPN
  n_frames = 1
  ret, frame = cap.read()
  while(n_frames < skipped_frames):
    ret, frame = cap.read()
    n_frames += 1
  if ret == False: 
    exit()
      
  [cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)
  # tracker init
  target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
  state = SiamRPN_init(frame, target_pos, target_sz, net)
  return state

# image and init box

cap = open_source_video(image_source)

#image_files = sorted(glob.glob('./human5/*.jpg'))
#init_rbox = [326.0, 419.0, 338.0, 420.0, 336.0, 453.0, 326.0, 453.0] # human5
#image_files = sorted(glob.glob('./bag/*.jpg'))
#init_rbox = [334.02,128.36,438.19,188.78,396.39,260.83,292.23,200.41] # bag
init_rbox = [16.0, 445.0, 158.0, 445.0, 158.0, 715.0, 16.0, 715.0] # rajeev

state = init_SiamRPN(cap, init_rbox, 2)

# tracking and visualization
toc = 0

n_frames = 0
while(True):
  ret, frame = cap.read()
  n_frames += 1
  if ret == False: 
    break
  tic = cv2.getTickCount()
  state = SiamRPN_track(state, frame)  # track
  toc += cv2.getTickCount()-tic
  res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
  res = [int(l) for l in res]
  cv2.rectangle(frame, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
  cv2.imshow('SiamRPN', frame)
  cv2.waitKey(1)

print('Tracking Speed {:.1f}fps'.format((n_frames)/(toc/cv2.getTickFrequency())))
