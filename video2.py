import numpy as np
import cv2 as cv
cap = cv.VideoCapture('2022-03-22_23-10-51.mp4')

# take first frame of the video
ret, frame = cap.read()
# setup initial location of window
x, y, width, height = 901, 501, 1008-901, 606-501
track_window = (x, y ,width, height)

# set up the ROI for tracking
roi = frame[y:y+height, x : x+width]

# hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
# mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255)))
#
# roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
# cv.normalize(roi_hist, roi_hist, 0, 255,cv.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
cv.imshow('roi',roi)

writer= cv.VideoWriter('basicvideo.mp4', cv.VideoWriter_fourcc(*'mp4v'), 20, (500,500))
while(1):
    ret, frame = cap.read()
    if ret == True:

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        ret,dst = cv.threshold(hsv,100,255,cv.THRESH_BINARY_INV)
        kernel = np.ones((5, 5), np.uint8)
        dst = cv.erode(dst, kernel, iterations=3)
        dst = cv.dilate(dst, kernel, iterations=5)


        # apply meanshift to get the new location
        _, track_window = cv.CamShift(dst, track_window, term_crit)

        # Draw it on image
        # pts = cv.boxPoints(ret)
        # # print(pts)
        # pts = np.int0(pts)
        # final_image = cv.polylines(frame, [pts], True, (0, 255, 0), 2)
        x,y,w,h = track_window
        final_image = cv.rectangle(frame, (x,y), (x+w, y+h), 255, 3)

        cv.imshow('dst', dst)
        cv.imshow('final_image',final_image)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break

cap.release()
writer.release()
cv.destroyAllWindows()
