import cv2
import mediapipe as mp
import math 
import time
import numpy as np
import json
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

exercise = None
child_id = None
start_time = None
max_duration = None
distance_threshold_close = None
distance_threshold_far = None

# Function to calculate Euclidean distance
def calculate_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# Define exercises with their respective distance thresholds
exercises = {
    "raise_weight": {"close": 0.30, "far": 0.50},
    "shoulder_shrugs": {"close": 0.10, "far": 0.11},
    "arm_cycle": {"close": 0.35, "far": 0.55},
    "wall_push": {"close": 0.10, "far": 0.25},
    "front_raise": {"close": 0.15, "far": 0.50}
}

# Function to save exercise data
def save_exercise_data(child_id, exercise, full_cycles, avg_time_per_cycle, accuracy):
    data = {
        "child_id": child_id,
        "count": full_cycles,
        "avg_time_per_cycle": avg_time_per_cycle,
        "accuracy": accuracy
    }

    file_path = "exercise.json"

    # Load existing data if the file exists
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            try:
                existing_data = json.load(file)
                if not isinstance(existing_data, dict):
                    existing_data = {}  # Ensure it's a dictionary
            except json.JSONDecodeError:
                existing_data = {}  # Handle empty or corrupt file
    else:
        existing_data = {}

    # Initialize exercise data if not present
    if exercise not in existing_data:
        existing_data[exercise] = {}

    # Add or update the data for the specific child and exercise
    existing_data[exercise][child_id] = data

    # Save the updated data back to the file
    with open(file_path, "w") as file:
        json.dump(existing_data, file, indent=4)

    #print(f"Data saved for {child_id} under exercise '{exercise}': {data}")  # Debugging log


def generate_video_feed():
    cap = cv2.VideoCapture(0)
    full_cycles = 0
    state = "close"
    cycle_times = []
    global start_time , exercise

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
            landmarks = results.pose_landmarks.landmark

            try:
                if exercise == "wall_push":
                    point1 = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    point2 = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    point3 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    point4 = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                else:
                    point1 = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    point2 = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    point3 = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    point4 = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                distance_right = calculate_distance(point1, point2)
                distance_left = calculate_distance(point3, point4)
                

                if state == "close" and distance_right >= distance_threshold_far and distance_left >= distance_threshold_far:
                    state = "far"
                elif state == "far" and distance_left <= distance_threshold_close and distance_right <= distance_threshold_close:
                    state = "close"
                    full_cycles += 1
                    cycle_times.append(time.time() - start_time)
                    start_time = time.time()
            except IndexError:
                pass

        avg_time_per_cycle = np.mean(cycle_times) if cycle_times else 0
        accuracy = min(100, (full_cycles / max(1, len(cycle_times))) * 100)

        cv2.putText(image, f'Exercise: {exercise}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f'Cycle Count: {full_cycles}', (50, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #cv2.putText(image, f'Avg Time: {avg_time_per_cycle:.2f}s', (50, 90), 
                    #cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        #cv2.putText(image, f'Accuracy: {accuracy:.2f}%', (50, 130), 
                    #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        save_exercise_data(child_id, exercise, full_cycles, avg_time_per_cycle, accuracy)


        _, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    

