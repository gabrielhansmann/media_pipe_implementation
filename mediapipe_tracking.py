import os
from datetime import datetime
import cv2
import mediapipe as mp
import json

# Initialize Mediapipe solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
pose = mp_pose.Pose(
    static_image_mode=False, model_complexity=1,
    enable_segmentation=True, min_detection_confidence=0.8,
    min_tracking_confidence=0.8)
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2,
    model_complexity=1, min_detection_confidence=0.8,
    min_tracking_confidence=0.8)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    refine_landmarks=True, min_detection_confidence=0.8,
    min_tracking_confidence=0.8)

# Set up video capture
cap = cv2.VideoCapture(0)  # Adjust the camera index if needed
landmark_data = []  # To store 3D landmarks for each frame

# Define output folder and ensure it exists
output_folder = "output_videos"
os.makedirs(output_folder, exist_ok=True)

# Generate unique filenames with timestamps
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_video_filename = os.path.join(output_folder, f"annotated_output_{timestamp}.avi")
output_json_filename = os.path.join(output_folder, f"landmarks_{timestamp}.json")

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter(output_video_filename, fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Initialize dictionary to store landmarks for this frame
    frame_landmarks = {'pose': [], 'hands': [], 'face': []}

    # Pose detection
    pose_results = pose.process(rgb_frame)
    if pose_results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        frame_landmarks['pose'] = [
            {'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in pose_results.pose_landmarks.landmark
        ]

    # Hand detection
    hands_results = hands.process(rgb_frame)
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            frame_landmarks['hands'].append([
                {'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks.landmark
            ])

    # Face mesh detection
    face_results = face_mesh.process(rgb_frame)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
            frame_landmarks['face'] = [
                {'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in face_landmarks.landmark
            ]

    # Append frame landmarks to the main data list
    landmark_data.append(frame_landmarks)

    # Save frame with overlay to video
    out_video.write(frame)
    cv2.imshow("Mediapipe Tracking", frame)
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

# Release resources
cap.release()
out_video.release()
cv2.destroyAllWindows()

# Save landmark data to JSON for Blender import
with open(output_json_filename, 'w') as f:
    json.dump(landmark_data, f)

print(f"Video saved as {output_video_filename}")
print(f"Landmark data saved as {output_json_filename}")