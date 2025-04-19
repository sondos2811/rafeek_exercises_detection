import cv2
import mediapipe as mp
import math
import time
import json
import os


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

start_time = None
time_per_count = []
correct_frames = 0
total_frames = 0
tolerance = 10  # Angle tolerance in degrees
exercise = None
side = None
child_id = id
avg_time = 0 

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


# Function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Function to calculate the angle between three points (for the elbow)
def calculate_angle(point1, point2, point3):
    vector1 = (point1[0] - point2[0], point1[1] - point2[1])
    vector2 = (point3[0] - point2[0], point3[1] - point2[1])
    
    dot_product = vector1[0]*vector2[0] + vector1[1]*vector2[1]
    magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
    
    if magnitude1 * magnitude2 == 0:
        return 0
    angle_rad = math.acos(dot_product / (magnitude1 * magnitude2))
    return math.degrees(angle_rad)

# Function to calculate the 360° angle of the wrist relative to the shoulder.
def calculate_360_angle(shoulder, wrist , side):
    dx = wrist[0] - shoulder[0]
    dy = shoulder[1] - wrist[1]  # Invert y for correct orientation

    if side == "left":  
        dx = -dx  # Flip x direction for the right side

    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    if angle_deg < 0:
        angle_deg += 360
    return angle_deg

# Define exercise parameters.
exercises = {
    "bilateral_contraction": {
        "start_elbow_angle": 170, 
        "end_elbow_angle": 70, 
        "min_wrist_360": 140, 
        "max_wrist_360": 180
    },
    "forearm_rotation": {
        "start_elbow_angle": 90, 
        "end_elbow_angle": 50, 
        "min_wrist_360": 305, 
        "max_wrist_360": 320
    },
    "stretching_exercise": { #done / right / left
        "start_elbow_angle": 155, 
        "end_elbow_angle": 110, 
        "min_wrist_360": 180, 
        "max_wrist_360": 200
    },
    "triceps_extension": { #done / right 
        "start_elbow_angle": 90, 
        "end_elbow_angle": 45, 
        "min_wrist_360": 105,
        "max_wrist_360": 135 
    },
}


def generate_video_feed():
    params = exercises[exercise]
    start_elbow_angle = params["start_elbow_angle"]
    end_elbow_angle = params["end_elbow_angle"]
    min_wrist_360 = params["min_wrist_360"]
    max_wrist_360 = params["max_wrist_360"]
    count = 0
    cycle_started = False
    accuracy = 0
    avg_time = 0  
      

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            h, w, _ = frame.shape
            landmarks = results.pose_landmarks.landmark

                # Select landmarks based on the chosen side
            if side == "left":
                shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                                int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
                elbow = (int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * w),
                             int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * h))
                wrist = (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * w),
                             int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * h))
            else:
                shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                                int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
                elbow = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w),
                             int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h))
                wrist = (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * w),
                             int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * h))

            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            wrist_360_angle = calculate_360_angle(shoulder, wrist ,side)
            distance = calculate_distance(shoulder, wrist)

                # Draw landmarks and lines
            cv2.circle(frame, shoulder, 10, (255, 0, 0), -1)
            cv2.circle(frame, elbow, 10, (0, 255, 0), -1)
            cv2.circle(frame, wrist, 10, (0, 0, 255), -1)
            cv2.line(frame, shoulder, elbow, (255, 255, 255), 3)
            cv2.line(frame, elbow, wrist, (255, 255, 255), 3)
            cv2.line(frame, shoulder, wrist, (0, 255, 255), 2)

                # Display angles
                

                # Exercise tracking logic
            if elbow_angle >= start_elbow_angle and (min_wrist_360 <= wrist_360_angle <= max_wrist_360) and not cycle_started:
                cycle_started = True
                start_time = time.time()  # Start time for this count
            elif elbow_angle <= end_elbow_angle and cycle_started:
                count += 1
                cycle_started = False
                if start_time:
                    time_taken = time.time() - start_time  # Time per repetition
                    time_per_count.append(time_taken)

                # ✅ **Draw information on the frame**
            #cv2.putText(frame, f"Elbow Angle: {int(elbow_angle)} deg", (50, 50),
                            #cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            #cv2.putText(frame, f"Wrist 360 Angle: {int(wrist_360_angle)} deg", (50, 90),
                            #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, f'Exercise: {exercise}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Cycle Count: {count}", (50, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if time_per_count:
                avg_time = sum(time_per_count) / len(time_per_count)
                #cv2.putText(frame, f"Avg Time per Count: {avg_time:.2f}s", (50, 90),
                                #.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if total_frames > 0:
                accuracy = (correct_frames / total_frames) * 100
                #cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (50, 130),
                                #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            save_exercise_data(child_id, exercise, count, avg_time, accuracy)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

       

    