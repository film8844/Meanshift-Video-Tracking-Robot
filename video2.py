import numpy as np
import cv2 as cv
cap = cv.VideoCapture('2022-03-22_23-10-51.mp4')


# We need to check if camera
# is opened previously or not
if (cap.isOpened() == False):
    print("Error reading video file")

# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
result = cv.VideoWriter('result.mp4',cv.VideoWriter_fourcc(*'mp4v'),30, size)

# take first frame of the video
ret, frame = cap.read()
# setup initial location of window
x, y, width, height = 901, 501, 1008-901, 606-501
track_window = (x, y ,width, height)

# set up the ROI for tracking
roi = frame[y:y+height, x : x+width]


# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
cv.imshow('roi',roi)

ox , oy = int(x+width/2) , int(y+height/2)
mask = np.zeros_like(frame)
while(1):

    ret, frame = cap.read()
    if ret == True:

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)


        ret,dst = cv.threshold(hsv,80,255,cv.THRESH_BINARY_INV)
        kernel = np.ones((5, 5), np.uint8)
        dst = cv.morphologyEx(dst, cv.MORPH_CLOSE, np.ones((20, 20), np.uint8))
        dst = cv.erode(dst, kernel, iterations=3)

        dst = cv.dilate(dst, kernel, iterations=8)


        # apply meanshift to get the new location
        _, track_window = cv.CamShift(dst, track_window, term_crit)

        # Draw it on image
        # pts = cv.boxPoints(ret)
        # # print(pts)
        # pts = np.int0(pts)
        # final_image = cv.polylines(frame, [pts], True, (0, 255, 0), 2)
        x,y,w,h = track_window
        final_image = cv.rectangle(frame, (x,y), (x+w, y+h), 255, 3)
        final_image = cv.circle(final_image, (int(x+w/2) ,int(y+h/2) ), 10, (0,0,255), -1)

        mask = cv.line(mask, (ox,oy), (int(x+w/2) ,int(y+h/2)), (0, 255, 0), 2)
        ox,oy = int(x+w/2) ,int(y+h/2)

        final_image = cv.add(final_image, mask)
        # Using cv2.putText() method
        final_image = cv.putText(final_image, 'position: x={} y={}'.format(int(x+w/2)-600 ,int(y+h/2)), (50, 50), cv.FONT_HERSHEY_SIMPLEX,
                            1, (0,255, 0), 2, cv.LINE_AA)



        cv.imshow('dst', dst)
        cv.imshow('final_image',final_image)
        result.write(final_image)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break

cap.release()
result.release()
cv.destroyAllWindows()
