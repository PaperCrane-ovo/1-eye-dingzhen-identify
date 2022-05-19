import dlib
import cv2
import numpy as np
import sys

maxdist = 0
detector = dlib.get_frontal_face_detector()


def face_locator(img):
    '''
    脸部定位,如果有多张人脸则返回最大的人脸
    '''
    dets = detector(img, 0)
    if not dets:
        return None
    return max(dets,key=lambda d:d.area())#TODO


predictor = dlib.shape_predictor('res/shape_predictor_68_face_landmarks.dat')

def extract_features(img, face_loc):
    '''
    利用dlib的68点模型,提取特征
    '''
    landmark = predictor(img, face_loc)
    key_points = []
    for i in range(68):
        pos = landmark.part(i)
        #转换成np数组方便计算
        key_points.append(np.array([pos.x, pos.y],dtype = np.int32))
    return key_points


def cal(std_keypoints,self_keypoints):
    new_std=[]
    new_self=[]
    for i in range(68):
        '''
        将绝对坐标转换为相对坐标
        '''
        new_std.append(std_keypoints[i] - std_keypoints[0])
        new_self.append(self_keypoints[i] -self_keypoints[0])
    sum = 0
    for i in range(68):
        sum+=np.linalg.norm(new_std[i]-new_self[i])
    return sum



def draw(img,face_loc,dis=1):
    dist = 1-np.tanh(dis/10000)
    p1 = x1,y1 = face_loc.left(),face_loc.top()
    p2 = x2,y2 = face_loc.right(),face_loc.bottom()
    p3 = int((x1+x2)/2),y2+10
    cv2.rectangle(img,p1,p2,(0,0,255),2)
    if dis!=1:
        global maxdist 
        maxdist = max(maxdist,dist)
        cv2.putText(img,str(maxdist),(p3),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
    return img


def reprocess(std_faceloc,img,self_faceloc):
    std_width = std_faceloc.width()
    std_height = std_faceloc.height()
    self_width = self_faceloc.width()
    self_height = self_faceloc.height()
    new_img = cv2.resize(img,None,fx = std_width/self_width,fy = std_height/self_height,\
                        interpolation = cv2.INTER_LINEAR if std_height > self_height else cv2.INTER_AREA)
    return new_img

def main():
    std_path = ''
    self_path = ''
    if len(sys.argv ) == 3:
        std_path = sys.argv[1]
        self_path = sys.argv[2]
    elif len(sys.argv) == 2:
        std_path = sys.argv[1]
        self_path = 'self.jpg'
    else :
        std_path = 'res/std_dingzhen.jpg'
        self_path = 'res/self.jpg'
    std_img = cv2.imread(std_path)
    std_face_loc = face_locator(std_img)
    if std_face_loc:
        std_keypoints = extract_features(std_img, std_face_loc)
    cap = cv2.VideoCapture(0)
    n = 0
    img = None
    dis = 0
    global maxdist
    stimg = draw(std_img,std_face_loc)

    while True:
        
        if n==100: n=0

        ret,self_img = cap.read()
        self_img = cv2.flip(self_img,1)
        self_face_loc = face_locator(self_img)
        if self_face_loc:
            if not n:
                new_img = reprocess(std_face_loc,self_img,self_face_loc)
                new_face_loc = face_locator(new_img)
                self_keypoints = extract_features(new_img, new_face_loc)
                dis = cal(std_keypoints,self_keypoints)
            img = draw(self_img,self_face_loc,dis)
            cv2.imshow('standard!',stimg)
            cv2.imshow('myself',img)
            cv2.waitKey(1)
        else: maxdist = 0
        n+=1




def main2():
    std_img = cv2.imread('res/std_dingzhen.jpg')
    std_face_loc = face_locator(std_img)
    if std_face_loc:
        std_keypoints = extract_features(std_img, std_face_loc)
    self_img = cv2.imread('res/std_dingzhen1.jpg')
    self_face_loc = face_locator(self_img)
    if self_face_loc:
        new_img = reprocess(std_face_loc,self_img,self_face_loc)
        new_face_loc = face_locator(new_img)
        self_keypoints = extract_features(new_img, new_face_loc)
        dis = cal(std_keypoints,self_keypoints)
        std_im = draw(std_img,std_face_loc)
        img = draw(new_img,new_face_loc,dis)
        cv2.imshow('std',std_im)
        cv2.imshow('self',img)
        cv2.waitKey()



if __name__ == '__main__':
    main()