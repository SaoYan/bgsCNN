import numpy as np
import cv2
import libbgs
import os
import os.path
from utilities import *

def generate_bg(root_dir):
    cv2.namedWindow("frame")
    cv2.namedWindow("foregrond mask")
    cv2.namedWindow("groundtruth")
    cv2.namedWindow("background model")
    for __, dirnames_l0, __ in walklevel(root_dir, level = 0):
        for dirname_l0 in dirnames_l0:
            if not os.path.exists(root_dir + "/" + dirname_l0 + "/done") or os.path.isfile(root_dir + "/" + dirname_l0 + "/done"):
                print ("start dealing with " + dirname_l0)
                if not os.path.exists(root_dir + "/" + dirname_l0 + "/bg") or os.path.isfile(root_dir + "/" + dirname_l0 + "/bg"):
                    os.makedirs(root_dir + "/" + dirname_l0 + "/bg")
                num = 1
                bgs = libbgs.SuBSENSE()
                F = open(root_dir + "/" + dirname_l0 + "/temporalROI.txt", 'r')
                line  = F.read().split(' ')
                begin = int(line[0]); end = int(line[1])
                ROI_mask = cv2.imread(root_dir + "/" + dirname_l0 + "/ROI.bmp")
                while True:
                    frame_filename = root_dir + "/" + dirname_l0 + "/input/" + num2filename(num, "in") + ".jpg"
                    gt_filename = root_dir + "/" + dirname_l0 + "/groundtruth/" + num2filename(num, "gt") + ".png"
                    frame = cv2.imread(frame_filename)
                    gt = cv2.imread(gt_filename)
                    check = (frame[:,:,0] == frame[:,:,1])
                    if check.all():
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    fg = cv2.bitwise_and(bgs.apply(frame), ROI_mask[:,:,0])
                    bg_model = bgs.getBackgroundModel()
                    addition = 83*np.ones(fg.shape, dtype=np.uint8)
                    fg_mask = cv2.add(fg, addition, mask=cv2.bitwise_not(ROI_mask[:,:,0]))
                    fg_mask = cv2.add(fg_mask, fg)
                    cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
                    cv2.putText(frame, str(num), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
                    cv2.imshow("frame", frame)
                    cv2.imshow("foregrond mask", fg_mask)
                    cv2.imshow("groundtruth", gt)
                    cv2.imshow("background model", bg_model)
                    if (num >= begin) & (num <= end):
                        bg_filename = root_dir + "/" + dirname_l0 + "/bg/" + num2filename(num, "bg") + ".jpg"
                        cv2.imwrite(bg_filename, bg_model)
                    num = num + 1
                    if num > end:
                        print ("finish with " + dirname_l0)
                        break
                    cv2.waitKey(20)
                os.makedirs(root_dir + "/" + dirname_l0 + "/done")
