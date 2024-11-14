import os
from datetime import datetime
import cv2
import mediapipe as mp
import json

# Initialize Mediapipe solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# Set capture resolution and frame rate
capture_width, capture_height = 1920, 1080  # Set to full HD
frame_rate = 30  # Increase frame rate for smoother output

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
cap.set(cv2.CAP_PROP_FPS, frame_rate)

# Define output folder
output_folder = "output_videos"
os.makedirs(output_folder, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
raw_video_filename = os.path.join(output_folder, f"raw_capture_{timestamp}.avi")

# Save captured video for post-processing
fourcc = cv2.VideoWriter_fourcc(*'XVID')
raw_video = cv2.VideoWriter(raw_video_filename, fourcc, frame_rate, (capture_width, capture_height))

# Capture video first
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    raw_video.write(frame)
    cv2.imshow("Capturing Video", frame)
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to stop recording
        break

# Release raw capture resources
cap.release()
raw_video.release()
cv2.destroyAllWindows()

print(f"Raw video saved as {raw_video_filename}")

# Post-processing phase
landmark_data = []
output_video_filename = os.path.join(output_folder, f"annotated_output_{timestamp}.avi")
output_json_filename = os.path.join(output_folder, f"landmarks_{timestamp}.json")
post_cap = cv2.VideoCapture(raw_video_filename)

# Set up video writer for annotated video
out_video = cv2.VideoWriter(output_video_filename, fourcc, frame_rate, (capture_width, capture_height))

# More complex model settings
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True)
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Process each frame of the saved video
while post_cap.isOpened():
    ret, frame = post_cap.read()
    if not ret:
        break
    
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Initialize dictionary to store landmarks for this frame
    frame_landmarks = {'pose': [], 'hands': [], 'face': []}

    # Pose detection
    pose_results = pose.process(rgb_frame)
    if pose_results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        frame_landmarks['pose'] = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in pose_results.pose_landmarks.landmark]

    # Hand detection
    hands_results = hands.process(rgb_frame)
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            frame_landmarks['hands'].append([{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks.landmark])

    # Face mesh detection
    face_results = face_mesh.process(rgb_frame)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
            frame_landmarks['face'] = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in face_landmarks.landmark]

    # Append frame landmarks to the main data list
    landmark_data.append(frame_landmarks)

    # Save frame with overlay to video
    out_video.write(frame)

# Release resources
post_cap.release()
out_video.release()

# Save landmark data to JSON
with open(output_json_filename, 'w') as f:
    json.dump(landmark_data, f)

print(f"Annotated video saved as {output_video_filename}")
print(f"Landmark data saved as {output_json_filename}")
