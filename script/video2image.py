import cv2
import os
import shutil

def run(video_file_name, image_format,
                train_timeF, train_offset,
                valid_timeF, valid_offset,
                test_timeF, test_offset,
		train_directrory, valid_directrory, test_directrory):

    if os.path.exists(train_directrory):
        shutil.rmtree(train_directrory)
    if os.path.exists(valid_directrory):
        shutil.rmtree(valid_directrory)
    if os.path.exists(test_directrory):
        shutil.rmtree(test_directrory)
    os.mkdir(train_directrory)
    os.mkdir(valid_directrory)
    os.mkdir(test_directrory)
    vc = cv2.VideoCapture(video_file_name) #读入视频文件
    c=0
    rval=vc.isOpened()

    while rval:   #循环读取视频帧
        c = c + 1
        rval, frame = vc.read()
        if(c % train_timeF == train_offset): #每隔timeF帧进行存储操作
            cv2.imwrite('{}/{}.{}'.format(train_directrory, c, image_format), frame) #存储为图像
        elif(c % valid_timeF == valid_offset): #每隔timeF帧进行存储操作
            cv2.imwrite('{}/{}.{}'.format(valid_directrory, c, image_format), frame) #存储为图像
        elif(c % test_timeF == test_offset): #每隔timeF帧进行存储操作
            cv2.imwrite('{}/{}.{}'.format(test_directrory, c, image_format), frame) #存储为图像
    vc.release()


