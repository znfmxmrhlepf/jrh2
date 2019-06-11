import random
import time
import cv2
import tools
import glob
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np

ImageDir = './VideoFrames/3/*.jpg'
imgList = sorted(glob.glob(ImageDir), key=lambda path:int(os.path.basename(path)[:-4]))
col = 'g'
cnt = 1

tf.enable_eager_execution()
model1 = tools.get_model1()
model2 = tools.get_model2()


ckpt = tf.train.Checkpoint(model=model1)
ckpt.restore(tf.train.latest_checkpoint('./data/1'))
ckpt = tf.train.Checkpoint(model=model2)
ckpt.restore(tf.train.latest_checkpoint('./data/4'))

spf = 0.03346
stat = tools.Status(spf)
print('\n\n\n')
flag = False 
img = None
for i, imgDir in enumerate(imgList):

    img = cv2.imread(imgDir)
    img = cv2.resize(img, (1200, 900), interpolation=cv2.INTER_AREA)
    cv2.imshow(',', img)
    cv2.waitKey(1) 
    if i%7: continue

    flatImg = tools.getFlatImg(img, col)
    if type(flatImg) == type(-1):
        flag = stat.push(set([-1]), i)
        continue
  
    circImg, circList = tools.hough_circle_transform(flatImg)
  
    if type(circImg) == type(-1):
        flag = stat.push(set([-1]), i)
        continue

    book_exist = []
    
    for circ in circList:
        circ = cv2.resize(circ, (30, 30), interpolation=cv2.INTER_AREA)
        label = np.argmax(model2(tf.expand_dims(tf.cast(circ/255, tf.float32), 0)).numpy())
        if label != 0:
            book_exist.append(label)
    
    flag = stat.push(set(sorted(book_exist)), i)
   
