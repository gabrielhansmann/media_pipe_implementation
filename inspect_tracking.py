import os
import cv2
import mediapipe as mp
import json
import time
import numpy as np
from numpy import matmul

# Initialize Mediapipe solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# Define input and output folders
input_folder = "input_videos"
output_folder = "output_videos"
os.makedirs(output_folder, exist_ok=True)

# Process each video in the input folder
for video_file in os.listdir(input_folder):
    if not video_file.endswith(('.mp4', '.avi', '.mov')):
        continue

    input_video_path = os.path.join(input_folder, video_file)
    base_filename = os.path.splitext(video_file)[0]
    output_video_filename = os.path.join(output_folder, f"annotated_{base_filename}.avi")
    output_video_skeleton_filename = os.path.join(output_folder, f"annotated_skeleton_{base_filename}.avi")
    output_json_filename = os.path.join(output_folder, f"landmarks_{base_filename}.json")

    # Open the input video
    post_cap = cv2.VideoCapture(input_video_path)
    capture_width = int(post_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    capture_height = int(post_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(post_cap.get(cv2.CAP_PROP_FPS))

    # Set up video writer for annotated video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(output_video_filename, fourcc, frame_rate, (capture_width, capture_height))

    # Set up video writer for skeleton-only video
    out_video_skeleton = cv2.VideoWriter(output_video_skeleton_filename, fourcc, frame_rate, (capture_width, capture_height))

    # More complex model settings
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True, smooth_landmarks=True, smooth_segmentation=True)  # smooth_landmarks enable_segmentation smooth_segmentation for lower jitter
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    # Initialize list to store landmark data
    landmark_data = []

    # Get the total number of frames in the video
    total_frames = int(post_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Start timer
    start_time = time.time()

    # Process each frame of the input video
    while post_cap.isOpened():
        ret, frame = post_cap.read()
        if not ret:
            break

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a black background frame for the skeleton
        skeleton_frame = np.zeros_like(frame)

        # Initialize dictionary to store landmarks for this frame
        frame_landmarks = {'pose': [], 'hands': [], 'face': []}

        t = time.time() / 10
        t = 2.1413 / 2
        # Pose detection
        pose_results = pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            for i, lm in enumerate(pose_results.pose_landmarks.landmark):
                if ((lm.HasField("visibility") and lm.visibility < 0.5) or (lm.HasField("presence") and lm.presence < 0.5)): continue
                pos = np.array([lm.x, lm.y, lm.z])
                c = np.cos(t)
                s = np.sin(t)
                d = np.array([-.5, -0.5, 0])
                rot_m = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
                pos_t = matmul(pos +d, rot_m) -d
                pose_results.pose_landmarks.landmark[i].x = pos_t[0]
                pose_results.pose_landmarks.landmark[i].y = pos_t[1]
                pose_results.pose_landmarks.landmark[i].z = pos_t[2]

            mp.solutions.drawing_utils.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(skeleton_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            frame_landmarks['pose'] = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in pose_results.pose_landmarks.landmark]

        # Hand detection
        hands_results = hands.process(rgb_frame)
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                for i, lm in enumerate(hand_landmarks.landmark):
                    if ((lm.HasField("visibility") and lm.visibility < 0.5) or (lm.HasField("presence") and lm.presence < 0.5)): continue
                    pos = np.array([lm.x, lm.y, lm.z])
                    print(f'Z-Depth: {lm.z}\n')
                    c = np.cos(t)
                    s = np.sin(t)
                    d = np.array([-.5, -0.5, 0])
                    rot_m = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
                    pos_t = matmul(pos +d, rot_m) -d
                    hand_landmarks.landmark[i].x = pos_t[0]
                    hand_landmarks.landmark[i].y = pos_t[1]
                    hand_landmarks.landmark[i].z = pos_t[2]
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                mp.solutions.drawing_utils.draw_landmarks(skeleton_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                frame_landmarks['hands'].append([{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks.landmark])

        # Face mesh detection
        face_results = face_mesh.process(rgb_frame)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                mp.solutions.drawing_utils.draw_landmarks(skeleton_frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                frame_landmarks['face'] = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in face_landmarks.landmark]

        # Append frame landmarks to the main data list
        landmark_data.append(frame_landmarks)

        # Save frame with overlay to video
        out_video.write(frame)
        out_video_skeleton.write(skeleton_frame)

        # Show elapsed time every second
        elapsed_time = time.time() - start_time
        if int(elapsed_time) % 1 == 0:
            print(f"\rGenerating... Elapsed time: {int(elapsed_time)} seconds", end='')

    # Calculate total elapsed time
    total_elapsed_time = time.time() - start_time

    # Release resources
    post_cap.release()
    out_video.release()
    out_video_skeleton.release()

    # Save landmark data to JSON
    with open(output_json_filename, 'w') as f:
        json.dump(landmark_data, f)

    print(f"\nAnnotated video saved as {output_video_filename}")
    print(f"Annotated skeleton-only video saved as {output_video_skeleton_filename}")
    print(f"Landmark data saved as {output_json_filename}")
    print(f"Time taken for processing {video_file}: {total_elapsed_time:.2f} seconds")
