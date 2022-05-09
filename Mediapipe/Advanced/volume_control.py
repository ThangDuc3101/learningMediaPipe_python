############################################ IMPORT LIB ##############################################################
import cv2
import mediapipe as mp
import time
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
########################################### INITIALIZE ###############################################################
##################### INITIALIZE MEDIAPIPE HAND DETECTION MODEL ######################
mp_hand = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hand = mp_hand.Hands(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)
########################## INITIALIZE OBJ FOR PYCAW ##################################
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
VolRange = volume.GetVolumeRange()
minVol = VolRange[0]
maxVol = VolRange[1]
############################################### LOAD CAM ##############################################################
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    start = time.time()
    position = []
################################################# PROCESS #############################################################    
    frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hand.process(frameRGB)
############################# HAND DETECTION ############################################    
    if results.multi_hand_landmarks:
        for detection in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame,detection,mp_hand.HAND_CONNECTIONS)
########################### GET THUMP TIP & INDEX FINGER TIP #############################          
            for id,lm in enumerate(detection.landmark):
                width, height ,channel = frame.shape
                cx,cy = int(lm.x*width),int(lm.y*height)
                if id ==4 or id==8:
                    position.append((id,cx,cy))                  
                    cv2.circle(frame,(cx,cy),5,(0,0,255),cv2.FILLED)
                if len(position) > 1:
                    cv2.line(frame,(position[0][1],position[0][2]),(position[1][1],position[1][2]),(0,0,255),2)
                    length = math.sqrt((position[0][1]-position[1][1])**2+(position[0][2]-position[1][2])**2)
########################### CONVERT LENGTH TO RANGE OF VOLUME #############################
                    setVol = np.interp(length,[5.0,200.0],[minVol,maxVol])
                    volume.SetMasterVolumeLevel(setVol, None)
############################################ FPS ###########################################                    
    end=time.time()
    fps=1.0/(end-start)
    cv2.putText(frame,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
    cv2.imshow("VOLUME CONTROL",frame)
############################################# QUIT CAM ################################################################    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()    