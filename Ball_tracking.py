import cv2
import numpy as np

video = cv2.VideoCapture(1)


# cv2.namedWindow('Track')
# cv2.resizeWindow('Track', 700, 512)


# def track(x):
#     pass
#
#
# cv2.createTrackbar('hue min', 'Track', 87, 179, track)
# cv2.createTrackbar('hue max', 'Track', 103, 179, track)
# cv2.createTrackbar('sat min', 'Track', 103, 255, track)
# cv2.createTrackbar('sat max', 'Track', 244, 255, track)
# cv2.createTrackbar('val min', 'Track', 63, 255, track)
# cv2.createTrackbar('val max', 'Track', 213, 255, track)



while True:
    ret, frame = video.read()
    frame = cv2.flip(frame,+1)
    #
    # h_min = cv2.getTrackbarPos('hue min', 'Track')
    # h_max = cv2.getTrackbarPos('hue max', 'Track')
    # s_min = cv2.getTrackbarPos('sat min', 'Track')
    # s_max = cv2.getTrackbarPos('sat max', 'Track')
    # val_max = cv2.getTrackbarPos('val max', 'Track')
    # val_min = cv2.getTrackbarPos('val min', 'Track')
    # print(f'HUE MIN : {h_min} HUE MAX : {h_max} SAT MIN : {s_min} SAT MAX : {s_max} VAL MIN : {val_min} VAL MAX : {val_max}')
    HSV_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_value = np.array([3,118,144])
    # lower_value = np.array([h_min, s_min,val_min])
    upper_value = np.array([80,255,255])
    # upper_value = np.array([h_max,s_max, val_max])

    mask_img = cv2.inRange(HSV_frame,lower_value,upper_value)

    clean_img = cv2.morphologyEx(mask_img,cv2.MORPH_OPEN,kernel=np.ones((3, 3), np.uint8))

    res = cv2.bitwise_and(frame,frame,None,clean_img)

    contours, hierarchy = cv2.findContours(clean_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        count = contours[0]
        Moment = cv2.moments(count)
        Cx = int(Moment['m10']/Moment['m00'])
        Cy = int(Moment['m01']/Moment['m00'])
        text2 = 'Object Detected'
        text = 'Location of object:'+'('+str(Cx)+','+str(Cy)+')'
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,text,(220,450),font,0.9,(255,0,0),4)
        cv2.putText(frame,text2,(200,30),font,1,(0,0,255),4)
        cv2.drawContours(frame,count,-1,(0,225,0),3)
    cv2.imshow('Images',frame)
    cv2.imshow('Maskimg',mask_img)
    if cv2.waitKey(1) == ord('s'):
        break

video.release()
cv2.destroyAllWindows()




