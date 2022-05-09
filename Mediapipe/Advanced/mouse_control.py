################################################# IMPORT LIB #############################################################

import cv2
import mediapipe as mp
import time
import pyautogui ,sys
import numpy as np
import math

################################################# INITIALIZE ################################################################

mp_hand = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hand = mp_hand.Hands(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

################################################# LOAD VIDEO ################################################################

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    start = time.time()

############################################## PROCESS IMG ##########################################################

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand.process(frameRGB)
    pointerPos = []

############################################## DETECT ###############################################################

    if results.multi_hand_landmarks:
        for detection in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, detection, mp_hand.HAND_CONNECTIONS)
            for id,lm in enumerate(detection.landmark):
                w,h,c = frame.shape                    
                if id ==12 or id == 8:
                   cx,cy = int(w*lm.x),int(h*lm.y) 
                   pointerPos.append((id,cx,cy))

############################################## MOUSE CLICK ##########################################################
    
    if len(pointerPos)>1 :
        length = math.sqrt(((pointerPos[0][1]-pointerPos[1][1])**2)+((pointerPos[0][2]-pointerPos[1][2])**2))
        x,y = int(pointerPos[0][1]*1920/480),int(pointerPos[0][2]*1080/640)
        pyautogui.moveTo(x,y)  
        print(length)    
        if length < 45.0:
            pyautogui.click(x,y)                

############################################## FPS #################################################################

    end = time.time()
    fps = 1.0/(end-start)
    cv2.putText(frame,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.imshow("MOUSE CONTROL",frame)
    
################################################# QUIT CAM ###############################################################
    
    if cv2.waitKey(10) & 0xFF ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()    