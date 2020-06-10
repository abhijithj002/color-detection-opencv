import cv2
import numpy as np


feed = cv2.VideoCapture(0)

"""while True:
	flag,frame = feed.read()
	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	#_, contours, _ =cv2.findCountours(feed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	color_detect(hsv)
	key = cv2.waitkey(1)
	if key == 27:
	    break
feed.release()
cv2.destroyAllWindows()
"""
def shape_detect(contours,frame):
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt,True),True)
        #cv2.drawContours(approx, cnt, 1,(0),5)
        if len(approx) == 4:
           x = approx.ravel()[0]
           y = approx.ravel()[1]
           cv2.putText(frame,"Square",(x,y), cv2.FONT_HERSHEY_COMPLEX , 1 ,(0))
        if len(approx) > 5:
           x = approx.ravel()[0]
           y = approx.ravel()[1]
           cv2.putText(frame,"Circle",(x,y), cv2.FONT_HERSHEY_COMPLEX , 1 ,(0))

def color_and_shape_detect(hsv,frame):

     r_filter = []
     low_red =np.array([30,80,80])
     high_red =np.array([179,255,255])
     red_mask = cv2.inRange(hsv,low_red,high_red)
     red = cv2.bitwise_and(frame,frame, mask=red_mask)
     red_contours, _ =cv2.findContours(red_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
     for rcont in red_contours:
         area1 = cv2.contourArea(rcont)
         if area1 > 2000:
             r_filter.append(rcont)
             cv2.drawContours(frame, r_filter,-1,(0,0,255),3)
             shape_detect(r_filter,frame)

 

     b_filter = []
     low_blue =np.array([94,80,2])
     high_blue =np.array([126,255,255])
     blue_mask = cv2.inRange(hsv,low_blue,high_blue)
     blue = cv2.bitwise_and(frame,frame, mask=blue_mask)
     blue_contours, _ =cv2.findContours(blue_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
     for bcont in blue_contours:
         area2 = cv2.contourArea(bcont)
         if area2 > 2000:
             b_filter.append(bcont)
             cv2.drawContours(frame, b_filter,-1,(255,0,0),3)
             shape_detect(b_filter,frame)
     
     
     g_filter=[]
     low_green =np.array([35,52,72])
     high_green =np.array([102,255,255])
     green_mask = cv2.inRange(hsv,low_green,high_green)
     green = cv2.bitwise_and(frame,frame, mask=green_mask)
     green_contours, _ =cv2.findContours(green_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
     for gcont in green_contours:
         area3 = cv2.contourArea(gcont)
         if area3 > 2000:
             g_filter.append(gcont)
             cv2.drawContours(frame, g_filter,-1,(0,255,0),3)
             shape_detect(g_filter,frame)



     cv2.imshow("FRAME",frame)


while True:
	flag,frame = feed.read()
	frame_blur = cv2.GaussianBlur(frame,(9,9),0)
	hsv = cv2.cvtColor(frame_blur,cv2.COLOR_BGR2HSV)
	color_and_shape_detect(hsv,frame_blur)
	key = cv2.waitKey(1)
	if key == 27:
            break
feed.release()
cv2.destroyAllWindows()


