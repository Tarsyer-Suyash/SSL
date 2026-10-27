import cv2
import time

def motion_detection(prev_frame, current_frame, MIN_CONTOUR_AREA=1000):


	# start_time_m = time.time()

	# compute the absolute difference between the current frame and first frame
	frameDelta = cv2.absdiff(prev_frame, current_frame)

	# convert it to grayscale, and blur it
	frameDelta = cv2.cvtColor(frameDelta, cv2.COLOR_BGR2GRAY)
	frameDelta = cv2.GaussianBlur(frameDelta, (5, 5), 0)
	ret_1, thresh = cv2.threshold(frameDelta, 50, 255, cv2.THRESH_BINARY)
	# print('motion Detector time >> ', (time.time() - start_time_m))

	# dilate the thresholded image to fill the holes, the find contours on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# if the length the contours tuple returned by cv2.findContours is '2' then
	# we are using either OpenCV v2.4, v4-beta or v4-official
	if len(cnts) == 2:
		cnts = cnts[0]

	# if the length of the contours tuple is '3' then we are using either OpenCV v3,
	# v4-pre or v4-alpha
	elif len(cnts) == 3:
		cnts = cnts[1]

	area = -1
	if len(cnts)>0:
		max_cnts = max(cnts, key=cv2.contourArea)
		area = cv2.contourArea(max_cnts)
		# print('AREA = {}'.format(area))
		if area > MIN_CONTOUR_AREA:
			return True, thresh, area

	return False, thresh, area
