import cv2
import mediapipe as mp
import time
import numpy as np

mp_draw = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
BG_COLOR = (192,192,192)
cap = cv2.VideoCapture(0)
with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as ss:
    while cap.isOpened():
        success, frame = cap.read()
        bg_image = None
        start = time.time()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = ss.process(frame)
        cv2.imshow("Mask",results.segmentation_mask)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        condition = np.stack((results.segmentation_mask,)*3,axis=-1)>0.15
        if bg_image is None:
            # Remove the background image
            # bg_image =np.zeros(frame.shape,dtype = np.uint8)
            # bg_image[:]=BG_COLOR
            # customize background image
            bg_image = cv2.imread("hand_landmarks.png")
            bg_image = cv2.resize(bg_image,(640,480))
            output_image = np.where(condition,frame,bg_image)
        end=time.time()
        fps=1.0/(end-start)
        cv2.putText(output_image,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow("Selfie Segmenttation", output_image)
        cv2.imshow("Original Image",frame)
        if cv2.waitKey(10) & 0xFF ==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()    