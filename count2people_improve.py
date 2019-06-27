# -*- coding: utf-8 -*-

##People counter
import numpy as np
import cv2
import Person
import time

cnt_up = 0
cnt_down = 0
count_up = 0
count_down = 0
state = 0

# Taking the video input
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("./counting_test.avi")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./output2.avi', fourcc, 20.0, (640, 480))

##cap.set(3,160) #Width
##cap.set(4,120) #Height

# Print the capture properties to console
for i in range(19):
    print (i, cap.get(i))

w = cap.get(3)
h = cap.get(4)
frameArea = h * w
areaTH = frameArea / 300
print ('Area Threshold', areaTH)

# Lines coordinate for counting
line_up = int(1.35 * (h / 5))
line_down = int(3.5 * (h / 5))

up_limit = int(.9 * (h / 5))
down_limit = int(4.5 * (h / 5))

print ("Red line y:", str(line_down))
print ("Blue line y:", str(line_up))
line_down_color = (255, 0, 0)
line_up_color = (0, 0, 255)
pt1 = [0, line_down]
pt2 = [w, line_down]
pts_L1 = np.array([pt1, pt2], np.int32)
pts_L1 = pts_L1.reshape((-1, 1, 2))
pt3 = [0, line_up]
pt4 = [w, line_up]
pts_L2 = np.array([pt3, pt4], np.int32)
pts_L2 = pts_L2.reshape((-1, 1, 2))

pt5 = [0, up_limit]
pt6 = [w, up_limit]
pts_L3 = np.array([pt5, pt6], np.int32)
pts_L3 = pts_L3.reshape((-1, 1, 2))
pt7 = [0, down_limit]
pt8 = [w, down_limit]
pts_L4 = np.array([pt7, pt8], np.int32)
pts_L4 = pts_L4.reshape((-1, 1, 2))

# Background Substractor
# fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
fgbg = cv2.createBackgroundSubtractorKNN()

# Structuring elements for morphographic filters
# kernelOp = np.ones((3,3),np.uint8)
# 1 16 20
# kernelOp = np.ones((7,7),np.uint8)
# kernelOp2 = np.ones((5,5),np.uint8)
# kernelCl = np.ones((11,11),np.uint8)
# 2
kernelCl2 = np.ones((3, 3), np.uint8)
kernelOp = np.ones((7, 7), np.uint8)
kernelCl = np.ones((17, 17), np.uint8)
kernelOp2 = np.ones((17, 17), np.uint8)
kernelCl3 = np.ones((8, 8), np.uint8)

# Variables
font = cv2.FONT_HERSHEY_SIMPLEX
persons = []
rect_co = []
max_p_age = 1
pid = 1
val = []

while (cap.isOpened()):
    ##for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    ret, frame = cap.read()
    if frame is None:
        break
    ##    frame = image.array

    # for i in persons:
    #     print i.age_one() #age every person one frame

    # Apply background subtraction
    fgmask = fgbg.apply(frame)
    fgmask2 = fgbg.apply(frame)

    # Binarization to eliminate shadows
    try:
        # 16 20 ----200
        ret, imBin = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
        ret, imBin2 = cv2.threshold(fgmask2, 127, 255, cv2.THRESH_BINARY)

        # #1---------KNN  会有光抖动噪声 需先剔除噪声   16 20
        # #Opening (erode->dilate) to remove noise. 去噪
        # mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
        # mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_OPEN, kernelOp)
        # #Closing (dilate -> erode) to join white regions. 膨胀--最好能使形状比较均匀
        # mask =  cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernelCl)
        # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernelCl)

        # 2-
        # closing 填补人衣服上的空缺
        imBin = cv2.morphologyEx(imBin, cv2.MORPH_CLOSE, kernelCl2)
        imBin2 = cv2.morphologyEx(imBin2, cv2.MORPH_CLOSE, kernelCl2)
        # Opening (erode->dilate) to remove noise. 去噪
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
        mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_OPEN, kernelOp)

        # Closing (dilate -> erode) to join white regions. 膨胀--最好能使形状比较均匀
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernelCl)
        # 腐蚀
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOp2)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernelOp2)
        # # Closing (dilate -> erode) to join white regions. 膨胀--最好能使形状比较均匀
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)
        # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernelCl)
        # # 腐蚀
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOp2)
        # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernelOp2)
        # # Closing (dilate -> erode) to join white regions. 膨胀--最好能使形状比较均匀
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)
        # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernelCl)
        # # 腐蚀
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOp2)
        # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernelOp2)
        # Closing (dilate -> erode) to join white regions. 膨胀--最好能使形状比较均匀
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl3)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernelCl3)




    except:
        print('EOF')
        print ('UP:', cnt_up)# + count_up
        print ('DOWN:', cnt_down)# + count_down
        break
    #################
    #   CONTOURS   #
    #################

    # RETR_EXTERNAL returns only extreme outer flags. All child contours are left behind.
    # _, contours0, hierarchy = cv2.findContours(mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    im2, contours0, hierarchy = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        rect = cv2.boundingRect(cnt)
        # print rect_co
        # if rect[2] > 100:
        #     if rect[1]!=0:
        #         rect_co.append(rect[1])
        #     if len(rect_co)>=2:
        #         if (rect_co[-1]-rect_co[-2]) > 0:
        #             count_down = rect[2]/60
        #             count_up = 0
        #             print 'down' ,count_down
        #         elif (rect_co[-1]-rect_co[-2]) < 0:
        #             count_up =  rect[2]/60
        #             count_down = 0
        #             print 'up',count_up

        #     continue
        area = cv2.contourArea(cnt)
        if area > areaTH:  # 矩形面积

            #################
            #   TRACKING    #
            #################

            # Missing conditions for multipersons, outputs and screen entries
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x, y, w, h = cv2.boundingRect(cnt)
            # print 'working'
            print (w)

            new = True
            if cy in range(up_limit, down_limit):
                for i in persons:
                    if abs(cx - i.getX()) <= w and abs(cy - i.getY()) <= h:
                        # 是在视频范围内的人（之前已经走进视频---是属于某个人-----基本上连续出现人的话都算是同一个人，只有没有人时才会清零）
                        # 其实没有准确的跟踪人，只是计数比较稳定？？？
                        # the object is close to one that has already been detected before
                        # print 'update'
                        new = False
                        i.updateCoords(cx, cy)  # update coordinates in the object and resets age
                        if i.going_UP(line_down, line_up) == True:

                            if w > 80:  # 没有意义？？？？减少误差？？？
                                count_up = w / 40
                                cnt_up += count_up
                                print ("记录了大于一个人:count_up:"+str(cnt_up))
                            else:
                                cnt_up += 1
                            print ("ID:", i.getId(), 'crossed going up at', time.strftime("%c"))
                        elif i.going_DOWN(line_down, line_up) == True:
                            if w > 80:  # 没有意义？？？？
                                count_down = w / 40
                                cnt_down += count_down
                                print ("记录了大于一个人:cnt_down:"+str(cnt_down))
                            else:
                                cnt_down += 1
                            print ("ID:", i.getId(), 'crossed going down at', time.strftime("%c"))
                        break
                    if i.getState() == '1':  # id为i的人已经计数过了
                        if i.getDir() == 'down' and i.getY() > down_limit:
                            i.setDone()
                        elif i.getDir() == 'up' and i.getY() < up_limit:
                            i.setDone()
                    if i.timedOut():  # true ----done=true
                        # get out of the people list
                        index = persons.index(i)
                        persons.pop(index)
                        del i  # free the memory of i
                if new == True:  # persons[]内没有人了（都删除/还没读）
                    p = Person.MyPerson(pid, cx, cy, max_p_age)
                    persons.append(p)
                    pid += 1  # 累计，之前删除的数组也算
            # new = True
            # print cy
            # if cy in range(up_limit,down_limit):
            #     for i in persons:
            #         if abs(cx-i.getX()) <= w and abs(cy-i.getY()) <= h:
            #             # the object is close to one that has already been detected before
            #             new = False
            #             i.updateCoords(cx,cy)
            #             val = i.getTracks()   #update coordinates in the object and resets age
            #             print val

            #     # print new
            #     if new == True:
            #         p = Person.MyPerson(pid,cx,cy, max_p_age)
            #         persons.append(p)
            #         pid += 1
            #     # print 'person length',len(persons)
            #     if len(val)>=2:
            #         if (val[-1][1]-val[-2][1]) > 0:
            #             cnt_down += 1;
            #             state='1'
            #             getdir = 'down'
            #             # print "ID:",i.getId(),'crossed going up at',time.strftime("%c")
            #         elif (val[-1][1]-val[-2][1]) < 0:
            #             cnt_up += 1;
            #             state = '1'
            #             getdir = 'up'
            #             # print "ID:",i.getId(),'crossed going down at',time.strftime("%c")
            #         val = []
            #         if state == '1':
            #                 if getdir == 'down':
            #                     done=True
            #                 elif getdir == 'up':
            #                     done = True
            #         if done:
            #              #get out of the people list
            #             j=persons[0]
            #             # print j
            #             index = persons.index(j)
            #             persons.pop(index)
            #             # print "delete"
            #             del j     #free the memory of i
            #################
            #   DRAWINGS     #
            #################
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.drawContours(frame, cnt, -1, (0,255,0), 3)

    # END for cnt in contours0

    #########################
    # DRAWING TRAJECTORIES  #
    #########################
    for i in persons:
        ##        if len(i.getTracks()) >= 2:
        ##            pts = np.array(i.getTracks(), np.int32)
        ##            pts = pts.reshape((-1,1,2))
        ##            frame = cv2.polylines(frame,[pts],False,i.getRGB())
        ##        if i.getId() == 9:
        ##            print str(i.getX()), ',', str(i.getY())
        # 标记人的id
        cv2.putText(frame, str(i.getId()), (i.getX(), i.getY()), font, 0.3, i.getRGB(), 1, cv2.LINE_AA)

    #################
    # DISPLAY ON FRAME    #
    #################
    str_up = 'UP: ' + str(cnt_up)# + count_up)
    str_down = 'DOWN: ' + str(cnt_down)# + count_down)
    frame = cv2.polylines(frame, [pts_L1], False, line_down_color, thickness=2)
    frame = cv2.polylines(frame, [pts_L2], False, line_up_color, thickness=2)
    frame = cv2.polylines(frame, [pts_L3], False, (255, 255, 255), thickness=1)
    frame = cv2.polylines(frame, [pts_L4], False, (255, 255, 255), thickness=1)
    cv2.putText(frame, str_up, (10, 40), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, str_up, (10, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, str_down, (10, 90), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, str_down, (10, 90), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    out.write(frame)
    cv2.imshow('Frame', frame)
    # cv2.imshow("subs",fgmask2)
    # cv2.imshow("dilate,erode", mask2)
    # cv2.imshow('Mask',mask)

    # Press ESC to exit
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
# END while(cap.isOpened())

#################
#   CLOSING    #
#################
print("总的进入人数为:")
print(str(int(cnt_up)))
print("总的出去人数为:")
print(str(int(cnt_down)))
cap.release()
cv2.destroyAllWindows()