import cv2
import mediapipe as mp
import time

mp_hand = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hand:
    while cap.isOpened():
        success, frame = cap.read()
        start = time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = hand.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #other options: multi_hand_world_landmarks, multi_handedness
        if results.multi_hand_landmarks:
            for detection in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame,detection,mp_hand.HAND_CONNECTIONS)
        end =time.time()
        fps = 1.0/(end-start)
        cv2.putText(frame, f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow("HAND_DETECTION",frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()    