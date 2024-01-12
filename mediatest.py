import cv2
from mediapipe import solutions
import time
from datetime import datetime
import multiprocessing
import numpy as np

def draw_landmarks_on_image(rgb_image, pose_landmarks):
    annotated_image = rgb_image.copy()

    if pose_landmarks:
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    
    return annotated_image

def camera_process(camera_index, shared_data):
    pose = solutions.pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=True, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(camera_index)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Camera {camera_index}: empty frame.")
            continue

        frame.flags.writeable = False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Store pose landmarks in the shared data structure
        if results.pose_landmarks:
            landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
            shared_data[camera_index] = landmarks

        frame.flags.writeable = True
        annotated_image = draw_landmarks_on_image(frame, results.pose_landmarks)

        cv2.imshow(f'Camera {camera_index}', annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

def display_skeletons(shared_data, landmark_names):
    while True:
        # Create an empty image for display
        skeleton_display = np.zeros((1000, 640, 3), dtype=np.uint8)  # Adjust the size as needed

        for idx, landmarks in shared_data.items():
            # Display each landmark with its name and coordinates rounded to three decimal places
            for i, lm in enumerate(landmarks):
                text = f"{landmark_names[i]}: ({lm[0]:.3f}, {lm[1]:.3f}, {lm[2]:.3f})"
                cv2.putText(skeleton_display, text, (10 + idx * 320, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Skeleton Data', skeleton_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    shared_data = manager.dict()

    landmark_names = [
        "Nose", "Left Eye Inner", "Left Eye", "Left Eye Outer", "Right Eye Inner", 
        "Right Eye", "Right Eye Outer", "Left Ear", "Right Ear", "Mouth Left", 
        "Mouth Right", "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
        "Left Wrist", "Right Wrist", "Left Pinky", "Right Pinky", "Left Index", 
        "Right Index", "Left Thumb", "Right Thumb", "Left Hip", "Right Hip", 
        "Left Knee", "Right Knee", "Left Ankle", "Right Ankle", "Left Heel", 
        "Right Heel", "Left Foot Index", "Right Foot Index"
    ]


    # Start camera processes
    process1 = multiprocessing.Process(target=camera_process, args=(0, shared_data))
    process2 = multiprocessing.Process(target=camera_process, args=(1, shared_data))

    # Start display process
    display_process = multiprocessing.Process(target=display_skeletons, args=(shared_data,landmark_names,))

    process1.start()
    process2.start()
    display_process.start()

    process1.join()
    process2.join()
    display_process.join()