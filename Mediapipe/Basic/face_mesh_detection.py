import cv2
import mediapipe as mp
import time

mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
drawing_spec = mp_draw.DrawingSpec

cap = cv2.VideoCapture(0)

with mp_face.FaceMesh(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)as face_mesh:
    while cap.isOpened():
        success, frame = cap.read()
        start = time.time()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = face_mesh.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for detection in results.multi_face_landmarks:
                #connections : FACEMESH_TESSELATION, FACEMESH_CONTOURS
                mp_draw.draw_landmarks(image=frame,
                                       connections = mp_face. FACEMESH_CONTOURS,
                                       landmark_list = detection,
                                       landmark_drawing_spec = drawing_spec,
                                       connection_drawing_spec = drawing_spec)
        end = time.time()
        fps = 1.0/(end-start)
        cv2.putText(frame, f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        cv2.imshow("FACE_MESH_DETECTION",frame)
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()    