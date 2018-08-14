#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 10:09:34 2018

@author: hans
"""

import sys
sys.path.append('.')
sys.path.append('/home/hans/caffe-ssd/python')
import common.tools_matrix as tools
import caffe
import cv2
import numpy as np
from operator import itemgetter
deploy = './mtcnn_models/12net.prototxt'
caffemodel = './mtcnn_models/12net.caffemodel'
net_12 = caffe.Net(deploy,caffemodel,caffe.TEST)
deploy = './mtcnn_models/24net.prototxt'
caffemodel = './mtcnn_models/24net.caffemodel'
net_24 = caffe.Net(deploy,caffemodel,caffe.TEST)
deploy = './mtcnn_models/48net.prototxt'
caffemodel = './mtcnn_models/48net.caffemodel'
net_48 = caffe.Net(deploy,caffemodel,caffe.TEST)
caffe.set_device(0)
caffe.set_mode_gpu()
def detectFace(img,threshold):
    caffe_img = (img.copy()-127.5)/128
    origin_h,origin_w,ch = caffe_img.shape
    scales = tools.calculateScales(img)
    out = []
    for scale in scales:
        hs = int(origin_h*scale)
        ws = int(origin_w*scale)
        scale_img = cv2.resize(caffe_img,(ws,hs))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_12.blobs['data'].reshape(1,3,ws,hs)
        net_12.blobs['data'].data[...]=scale_img
        out_ = net_12.forward()
        out.append(out_)
    image_num = len(scales)
    rectangles = []
    for i in range(image_num):    
        cls_prob = out[i]['prob1'][0][1]
        roi      = out[i]['conv4-2'][0]
        out_h,out_w = cls_prob.shape
        out_side = max(out_h,out_w)
        rectangle = tools.detect_face_12net(cls_prob,roi,out_side,1/scales[i],origin_w,origin_h,threshold[0])
        rectangles.extend(rectangle)
    rectangles = tools.NMS(rectangles,0.7,'iou')

    if len(rectangles)==0:
        return rectangles
    net_24.blobs['data'].reshape(len(rectangles),3,24,24)
    crop_number = 0
    for rectangle in rectangles:
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img,(24,24))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_24.blobs['data'].data[crop_number] =scale_img 
        crop_number += 1
    out = net_24.forward()
    cls_prob = out['prob1']
    roi_prob = out['conv5-2']
    rectangles = tools.filter_face_24net(cls_prob,roi_prob,rectangles,origin_w,origin_h,threshold[1])
    
    if len(rectangles)==0:
        return rectangles
    net_48.blobs['data'].reshape(len(rectangles),3,48,48)
    crop_number = 0
    for rectangle in rectangles:
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img,(48,48))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_48.blobs['data'].data[crop_number] =scale_img 
        crop_number += 1
    out = net_48.forward()
    cls_prob = out['prob1']
    roi_prob = out['conv6-2']
    pts_prob = out['conv6-3']
    rectangles = tools.filter_face_48net(cls_prob,roi_prob,pts_prob,rectangles,origin_w,origin_h,threshold[2])
    return rectangles    
def IOU(Reframe,GTframe):
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]-Reframe[0]
    height1 = Reframe[3]-Reframe[1]
    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]-GTframe[0]
    height2 = GTframe[3]-GTframe[1]
    endx = max(x1+width1,x2+width2)
    startx = min(x1,x2)
    width = width1+width2-(endx-startx)
    endy = max(y1+height1,y2+height2)
    starty = min(y1,y2)
    height = height1+height2-(endy-starty)
    if width <=0 or height <= 0:
        ratio = 0
    else:
        Area = width*height
        Area1 = width1*height1
        Area2 = width2*height2
        ratio = Area*1./(Area1+Area2-Area)
    return ratio
if __name__=='__main__':
    threshold = [0.6,0.6,0.7]
    cap = cv2.VideoCapture(0)
    pad=70
    LaFaceBox=[]
    CuFaceBoxList=[]
    NoseList=[]
    MeanFrame=5
    IOUthres=0.93
    while(cap.isOpened()):
        ret, frame = cap.read()
        k = cv2.waitKey(1)
        if k==27:break
        if frame is None:
            break
        frame =cv2.resize(frame,(640,480))
        rectangles = detectFace(frame,threshold)
        rectangles = sorted(rectangles, key=itemgetter(4),reverse=True)
        if len(rectangles)<1:
            LaFaceBox=[]
            cv2.imshow('frame', frame)
            print "No face"
            continue
        rectangle=rectangles[0]
        NoseList.append([int(rectangle[9]),int(rectangle[10])])
        while len(NoseList)>MeanFrame:
            NoseList.pop(0)
        NewNose = [0,0]
        for box in NoseList:
            NewNose[0]+=int(round(box[0]/float(len(NoseList))))
            NewNose[1]+=int(round(box[1]/float(len(NoseList))))
        NoseList.append([NewNose[0],NewNose[1]])
        CuFaceBox = [NewNose[0]-pad,NewNose[1]-pad,NewNose[0]+pad,NewNose[1]+pad]
        if len(LaFaceBox)<1:
            LaFaceBox=CuFaceBox
            continue
        IOU_rate = IOU(LaFaceBox,CuFaceBox)
        if IOU_rate>IOUthres:
            CuFaceBox=LaFaceBox
        LaFaceBox=CuFaceBox
        CuFaceBoxList.append([CuFaceBox[0],CuFaceBox[1]])
        while len(CuFaceBoxList)>MeanFrame:
            CuFaceBoxList.pop(0)
        NewFaceBox=[0,0]
        for box in CuFaceBoxList:
            NewFaceBox[0]+=int(round(box[0]/float(len(CuFaceBoxList))))
            NewFaceBox[1]+=int(round(box[1]/float(len(CuFaceBoxList))))
        CuFaceBoxList.append([NewFaceBox[0],NewFaceBox[1]])
        cv2.rectangle(frame,(int(NewFaceBox[0]),int(NewFaceBox[1])),(int(NewFaceBox[0]+pad*2),int(NewFaceBox[1]+pad*2)),(0,255,0),2)
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame', frame)
    cap.release()
    cv2.destroyAllWindows()
