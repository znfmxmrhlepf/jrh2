import tensorflow as tf
import glob
import cv2
import tools
import numpy as np

print('initializing...')
tf.logging.set_verbosity(tf.logging.ERROR)
vidPaths = './Video/5.mp4'
video = cv2.VideoCapture(vidPaths)
fps = video.get(cv2.CAP_PROP_FPS)
spf = 1 / fps
col = 'g'

tf.enable_eager_execution()
model1 = tools.get_model1()
model2 = tools.get_model2()

ckpt = tf.train.Checkpoint(model=model1)
ckpt.restore(tf.train.latest_checkpoint('./data/1'))
ckpt = tf.train.Checkpoint(model=model2)
ckpt.restore(tf.train.latest_checkpoint('./data/4'))

stat = tools.Status(spf)

i = 0
while 1:
    ret, img = video.read()
    if not ret: break
    i += 1
    if i%7:
        continue
    cv2.imshow('video', cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    cv2.waitKey(1) 
    tools.Process(img, col, stat, model2, i)

cv2.destroyAllWindows()
video.release()

print('done!')
