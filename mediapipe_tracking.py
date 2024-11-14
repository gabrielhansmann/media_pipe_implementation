import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

pose = mp_pose.Pose()
hands = mp_hands.Hands()
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)  # Adjust the camera index if needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Pose detection
    pose_results = pose.process(rgb_frame)
    if pose_results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Hand detection
    hands_results = hands.process(rgb_frame)
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Face detection
    face_results = face_mesh.process(rgb_frame)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

        cv2.imshow("Mediapipe Tracking", frame)
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()

