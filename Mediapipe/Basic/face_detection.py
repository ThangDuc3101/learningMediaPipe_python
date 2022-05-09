import cv2
import mediapipe as mp
import time

#init face_detection & drawing
mp_faceDetection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with mp_faceDetection.FaceDetection(min_detection_confidence = 0.5) as face_detection:
    while cap.isOpened():
        success, frame = cap.read()
        start = time.time()
        #convert to RGB image
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        #process and find
        result = face_detection.process(frame)
        
        if result.detections:
            for id, detection in enumerate(result.detections):
                mp_drawing.draw_detection(frame,detection)
                bBox = detection.location_data.relative_bounding_box
                width, height, c = frame.shape
                boundBox = int(bBox.xmin*width),int(bBox.ymin*height),int(bBox.width*width),int(bBox.height*height)
                cv2.putText(frame, f'{int(detection.score[0]*100)}%', (boundBox[0],boundBox[1]-20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        #reconvert to bgr image        
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)    
        end = time.time()
        fps = 1.0/(end-start)   
        cv2.putText(frame,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow("Face_detection_opencvPython_Mediapipe",frame)              
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()    