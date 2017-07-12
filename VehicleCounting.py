#! /usr/bin/python

import cv2
import sys
import copy
import numpy as np
import tensorflow as tf

def help():
    print ("--------------------------------------------------------------------------")
    print("Uasge:")
    print("python ./VehicleCounting.py -vid { <video filename> | 0 }")
    print("Example:")
    print("to use video file: python ./VehicleCounting.py -vid test.mp4")
    print("to use camera: python ./VehicleCounting.py -vid 0")
    print ("--------------------------------------------------------------------------\n")

def vehicle_location(detect_zone, W, L):
    '''''''''''''''''''''
    W: width of each lane
    L: width of the DVL
    '''''''''''''''''''''
    tmp_conv = list()
    tmp_1 = np.ones((L,W), dtype=np.float32)
    for c in range(detect_zone.shape[1]-W):
        conv_region = np.float32(detect_zone[0:L,c:c+W])
        S = np.sum(np.multiply(conv_region, tmp_1))
        tmp_conv.append(S)
    temp = np.array(tmp_conv)
    cv2.normalize(temp, temp, 0., 1. ,cv2.NORM_MINMAX)
    tmp_conv = temp.tolist()
    return tmp_conv

def dispHist(hist, histSize):
    maxVal = np.amax(np.array(hist))
    minVal = np.amin(np.array(hist))
    histDisp = np.zeros((histSize, histSize), dtype = np.uint8)
    hpt = int(0.9*histSize)
    if int(maxVal) != 0:
        for h in range(histSize):
            binVal = hist[h]
            intensity = int(binVal*hpt/maxVal)
            cv2.line(histDisp, (h,histSize), (h,histSize-intensity), (255))

    return histDisp

def processVideo(videoFilename):
    if videoFilename == "0":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(videoFilename)
    if not cap.isOpened():
        print("Unable to open video file: " + videoFilename)
        return
    else:
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # background subtractor
    pMOG = cv2.bgsegm.createBackgroundSubtractorMOG();
    # read input data & process
    # press'q' for quitting
    width_lane = 100; width_DVL = 100
    T_HDist = 60; T_VDist = 100; T_s = 0.3
    total_num = 0; add_num = 0
    peak_idx_current = list(); peak_idx_last = list()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to read next frame.")
            print("Exiting...")
            break
        # step 1: background subtraction
        mask = pMOG.apply(frame)
        # step 2: vehicle detection (morphology operation)
        objects = np.copy(mask)
        cross_element = cv2.getStructuringElement(cv2.MORPH_CROSS, (7,7))
        disk_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        objects = cv2.dilate(objects, cross_element)
        objects = cv2.dilate(objects, disk_element)
        # step 3: vehicle location
        detect_zone = np.copy(objects[height-24-width_DVL:height-24, 0:frame.shape[1]])
        tmp_conv = vehicle_location(detect_zone, width_lane, width_DVL)
        # step 4: vehicle counting
        num = 0; space = 20; min = 10000
        # detect all the peak candidates
        for i in range(len(tmp_conv)):
            if i < space:
                continue
            elif i > len(tmp_conv)-space: # // only compare with former elements
                if (tmp_conv[i] > T_s) & (tmp_conv[i] > tmp_conv[i-space]):
                    if len(peak_idx_current) == 0:
                        peak_idx_current.append(i)
                        num = num + 1
                    elif abs(i-peak_idx_current[num-1]) > T_HDist:
                        peak_idx_current.append(i)
                        num = num + 1
            else: # compare with both former and latter elements
                if (tmp_conv[i] > T_s) & (tmp_conv[i] > tmp_conv[i-space]) & (tmp_conv[i] > tmp_conv[i+space-1]):
                    if len(peak_idx_current) == 0:
                        peak_idx_current.append(i)
                        num = num + 1
                    elif abs(i-peak_idx_current[num-1]) > T_HDist:
                        peak_idx_current.append(i)
                        num = num + 1
        # counting
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == 1: # 1st frame
            add_num = len(peak_idx_current);
        else: # eliminate repeat counting
            for i in range(len(peak_idx_current)):
                for j in range(len(peak_idx_last)):
                    if abs(peak_idx_current[i]-peak_idx_last[j]) < min:
                        min = abs(peak_idx_current[i]-peak_idx_last[j])
                if min > T_VDist:
                    add_num = add_num + 1
                min = 10000
        total_num = total_num + add_num
        # find contours
        __, contours, __ = cv2.findContours(objects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hulls = list()
        for i in range(len(contours)):
            hulls.append(cv2.convexHull(contours[i]))
        # draw double virtual lines
        cv2.line(objects, (0, height-24), (width-1, height-24), (255,255,255), 2)
        cv2.line(objects, (0, height-24-width_DVL), (width-1, height-24-width_DVL), (255,255,255), 2)
        # draw hulls
        drawing = np.zeros(objects.shape, dtype = np.uint8);
        cv2.drawContours(drawing, hulls, -1, (255))
        # draw vehicle location hist
        histDisp = dispHist(tmp_conv, len(tmp_conv))
        for i in range(len(peak_idx_current)):
            cv2.line(histDisp, (peak_idx_current[i], 0), (peak_idx_current[i], histDisp.shape[0]-1), (0), 2)
        # write the frame number on the current frame
        numFrame = str(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv2.putText(frame, numFrame, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
        # write counting results on the current frame
        counting = "+" + str(add_num) + "   " + str(total_num)
        cv2.rectangle(frame, (10, 22), (100,40), (255,255,255), -1)
        cv2.putText(frame, counting, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
        # show
        cv2.imshow("Frame", frame)
        cv2.imshow("Vehicle Detection", objects)
        cv2.imshow("Contours", drawing)
        cv2.imshow("Vehicle Location", histDisp)
        # re-initialization
        add_num = 0
        peak_idx_last = copy.deepcopy(peak_idx_current)
        peak_idx_current = list()

        c = cv2.waitKey(30)
        if c >= 0:
            if chr(c) == 'q':
                break
            else:
                continue
        else:
            continue
    print ("total number of vehicles: " + str(total_num))
    cap.release()

def main():
    # print help information
    help()
    # check for the input parameter correctness
    if len(sys.argv) != 3:
        print("Incorret input list")
        print("exiting...")
        return
    # create GUI windows
    cv2.namedWindow("Frame")
    cv2.namedWindow("Vehicle Detection")
    cv2.namedWindow("Contours")
    cv2.namedWindow("Vehicle Location")
    # run algorithm
    if sys.argv[1] == "-vid":
        processVideo(sys.argv[2])
    else:
        print("Please, check the input parameters.")
        print("exiting...")
        return
    # destroy GUI windows
    cv2.destroyAllWindows();


if __name__ == "__main__":
    main()
