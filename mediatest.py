import cv2
from mediapipe import solutions
import time
from datetime import datetime


def draw_landmarks_on_image(rgb_image, pose_landmarks):
    annotated_image = rgb_image.copy()

    if pose_landmarks:
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    
    return annotated_image
# Initialize MediaPipe Pose.
pose = solutions.pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=True, min_detection_confidence=0.5)

# Capture video from the webcam.
cap = cv2.VideoCapture(0)

prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    new_frame_time = time.time()

    # To improve performance, optionally mark the frame as not writeable to pass by reference.
    frame.flags.writeable = False
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Draw the pose landmarks on the image.
    frame.flags.writeable = True
    annotated_image = draw_landmarks_on_image(frame, results.pose_landmarks)

    # Calculate FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps_text = f'FPS: {int(fps)}'

    # Get current time
    timestamp = datetime.now().strftime("%H:%M:%S:%MS")

    # Put timestamp and FPS on the frame
    cv2.putText(annotated_image, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(annotated_image, fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('MediaPipe Pose', annotated_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the windows.
cap.release()
cv2.destroyAllWindows()
pose.close()
