import time
import tensorflow as tf
import os
import glob
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import rawpy
import imageio

# plt.switch_backend('QT4Agg') not for colab
def imshow(img):
    if(len(img.shape)!=2):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

def imshow_max(img):
    if(len(img.shape)!=2):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

def imshow_time(img):
    if(len(img.shape)!=2):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show(block=False)
    plt.pause(3)
    plt.close()

def getRedArea(img):
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0+mask1

    # set my output img to zero everywhere except my mask
    img[np.where(mask==0)] = 0

    return img

def getYellowArea(img):
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([20,120,50])
    upper = np.array([30,255,255])
    mask = cv2.inRange(img_hsv, lower, upper)

    # set my output img to zero everywhere except my mask
    img[np.where(mask==0)] = 0

    return img

def getSqS(p):
    f = lambda x: abs(x)/2
    return f(ccw(p[0], p[1], p[3]) + ccw(p[1], p[2], p[3]))

def getGreenArea(img):
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([60,100,50])
    upper_green = np.array([95,255,255])

    mask = cv2.inRange(img_hsv, lower_green, upper_green)

    # set my output img to zero everywhere except my mask
    img[np.where(mask==0)] = 0

    return img

def getBlueArea(img):
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([100,120,50])
    upper = np.array([120,255,255])

    mask = cv2.inRange(img_hsv, lower, upper)

    # set my output img to zero everywhere except my mask
    img[np.where(mask==0)] = 0

    return img

def ccw(p1, p2, p3):
    ccw = (p2[0]-p1[0])*(p3[1]-p1[1])-(p2[1]-p1[1])*(p3[0]-p1[0])
    return ccw    

def getFlatImg(img, col):
    p1 = contour_approx(img.copy(), col)
    if type(p1)==type(-1):
        return -1
    p1 = np.float32(p1)
    ccwv = ccw(p1[0], p1[1], p1[2]) 
    
    if ccwv>0:
       p1=np.flip(p1, 0)
    
    minidx = 0
    val = 100000

    for i in range(4):
        if val > p1[i][0]+p1[i][1]:
            val = p1[i][0]+p1[i][1]
            minidx=i

    p1 = np.roll(p1, -2*minidx)
    p2 = np.array([[5, 5],[5, 995],[995, 995],[995, 5]],np.float32)

    # p1의 좌표에 표시. perspective 변환 후 이동 점 확인.
    
    m = cv2.getPerspectiveTransform(p1, p2)

    dst = cv2.warpPerspective(img, m, (1000,1000))
    
    return dst

def readDNG(path):
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

def find_all_polygon(img, col):
 
    if col == 'r':
        img = getRedArea(img) 
    if col == 'g':
        img = getGreenArea(img)
    if col == 'y':
        img = getYellowArea(img)
    if col == 'b':
        img = getBlueArea(img)

    img2 = img.copy()
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgray[np.where(imgray>0)] = 255
    edge = cv2.Canny(imgray, 100, 200)
    edge, contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    MAX_S=0
    MAX_approx=None

    for cnt in contours:
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
       
        cv2.drawContours(img, [approx], 0, (0,255,0),3)

    #imshow(img)

    return MAX_approx

def contour_approx(img, col):
    imgray = None
    if len(img.shape)!=2:
        if col == 'r':
            img = getRedArea(img)
        elif col == 'g':
            img = getGreenArea(img)
        elif col == 'y':
            img = getYellowArea(img)
        elif col == 'b':
            img = getBlueArea(img)
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    else:
        imgray = img.copy()

    # Morphology Opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    imgray = cv2.morphologyEx(imgray, cv2.MORPH_OPEN, kernel, iterations=2)
    edge = cv2.Canny(imgray, 100, 200)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    MAX_S=0
    MAX_approx=0
    for cnt in contours:
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = np.squeeze(cv2.approxPolyDP(cnt, epsilon, True))
        if len(approx)==4:
            s = getSqS(approx)
            if s > MAX_S:
                MAX_approx=approx
                MAX_S=s

    if MAX_S < 50000:
        return -1
    try:
        cv2.drawContours(img, [MAX_approx], 0, (0,255,0),2)
    except:
        return -1
    
    return MAX_approx

def morph_contour_approx(img):
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, \
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1)), iterations=1)
    img = cv2.resize(img, (600, 450), interpolation=cv2.INTER_AREA)
    contour_approx(img)

def blur(img, ksize):
    return cv2.blur(img, (ksize, ksize))

def binary_blur(img, col, ksize):
    img = img.copy()
    if col == 'r':
        img = getRedArea(img)
    if col == 'g':
        img = getGreenArea(img)
    if col == 'y':
        img = getYellowArea(img)
    if col == 'b':
        img = getBlueArea(img)
    img[np.where(img>0)]=255
    return blur(img, ksize)
    
def hough_circle_transform(img):
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 40, param1=40, param2=20, minRadius=15, maxRadius=21)

    circleset = []
    if type(circles) == type(None):
        return -1, circleset
    
    circles = np.int16(np.around(np.squeeze(circles, axis=0)))

    retImg = img.copy()
    for i in circles:
        circleset.append(img.copy()[i[1]-i[2]-2:i[1]+i[2]+3, i[0]-i[2]-2:i[0]+i[2]+3])
        cv2.circle(retImg,(i[0],i[1]),i[2]+2,(0,255,0),2)

    return img, circleset 

def load_data():
    # image totally 3179
    # test data 3001 ~ 3179 / train data 1 ~ 3000
    # 769   
    ImageDir = './Images/Circle/4/*.jpg'
    pathList = sorted(glob.glob(ImageDir),
            key=lambda path:int(os.path.basename(path)[:-4]))
    images = np.array([cv2.imread(path) for path in pathList])
    labels = np.array([int(label[:-1]) for label in open('./Images/Circle/4/label.txt').readlines()])
    p = np.random.permutation(len(images))
    images = images[p]
    labels = labels[p]

    print(images.shape)
    print(labels.shape)

    return images[:18000], labels[:18000], images[18000:], labels[18000:]

def get_model1():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', input_shape=(30, 30, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(0.3),


        tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(11, activation='softmax')
    ])

    return model

def get_model2():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', input_shape=(30, 30, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(11, activation='softmax')
    ])


    return model

class Status():
    def __init__(self, spf):
        self.human = False
        self.status = set()
        self.cur = []
        self.bef = []
        self.bbef = []
        self.spf = spf
        self.books = set()

    def push(self, books, i):
        self.bbef, self.bef, self.cur = self.bef, self.cur, books
        if self.bef == self.cur and self.bbef == self.bef and self.cur != self.status:
            text = str(round(self.spf * i, 2)) + '   '
            
            if self.cur == set([-1]):
                text += "Hello, there!"

            else:
                poped = self.books - self.cur
                if len(poped) == 0:
                    poped = 'Nothing'

                inserted = self.cur - self.books
                if len(inserted) == 0:
                    inserted = 'Nothing'

                text += str(poped) + ' poped / ' + str(inserted) + ' inserted'
            
                self.books = self.cur 

            self.status = self.cur

            return True, text

        return False, ''

def Process(img, col, stat, model, step):

#    imgp = img[:,240:1680]
#    imgp = cv2.resize(imgp, (1200, 900), interpolation=cv2.INTER_AREA)
    imgp = img
    flatImg = getFlatImg(imgp, col)
    
    if type(flatImg) == type(-1):
        flag, text = stat.push(set([-1]), step)
        if flag: print(text)
        return 

    circImg, circList = hough_circle_transform(flatImg)

    if type(circImg) == type(-1):
        flag, text = stat.push(set([-1]), step)
        if flag: print(text)
        return 

    book_exist = []
  
    for circ in circList:
        circ = cv2.resize(circ, (30, 30), interpolation=cv2.INTER_AREA)
        label = np.argmax(model(tf.expand_dims(tf.cast(circ/255, tf.float32), 0)).numpy())
        if label != 0:
            book_exist.append(label)
   
    flag, text = stat.push(set(sorted(book_exist)), step)
    if flag: print(text) 

    
if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    print(x_train.shape)
    imshow(x_train[10])
    print(y_train[10])
