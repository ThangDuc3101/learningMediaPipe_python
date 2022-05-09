import cv2
import mediapipe as mp
import time

mp_objectron = mp.solutions.objectron
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
# model_name : "Cup", "Shoe", "Chair", "Camera"
with mp_objectron.Objectron(max_num_objects=1, min_detection_confidence = 0.5, min_tracking_confidence = 0.5, model_name = "Shoe") as objectron:
    while cap.isOpened():
        success, frame = cap.read()
        start = time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = objectron.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.detected_objects:
            for detection in results.detected_objects:
                mp_draw.draw_landmarks(frame,detection.landmarks_2d,mp_objectron.BOX_CONNECTIONS)
                mp_draw.draw_axis(frame, detection.rotation, detection.translation)
        end = time.time()
        fps = 1.0/(end-start)
        cv2.putText(frame, f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),2)
        cv2.imshow("3D_OBJECT_DETECTION", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()    