import cv2
import mediapipe as mp
import numpy as np
import time
import math
import threading
import json
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

child_id = None
#exercise_active = False  # Track if exercise is being performed
start_time = time.time()
max_duration = 120 # 2 minutes in seconds
fingers_done = {'Index' , 'Middle' , 'Ring' , 'Pinky'}
exercise = None  
threshold_open=  0.5  
threshold_closed = 0.3
side = None

 
# Global variable to track the video processing thread
video_thread = None
video_thread_running = False  # To prevent multiple threads from starting

# Function to save exercise data
def save_exercise_data(child_id, exercise, side , done):
    data = {
        "child_id": child_id,
        "Side": side,
        "Done": done
        
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


# Function to calculate 360-degree angle of the thumb
def calculate_360_angle(wrist, thumb_base, thumb_tip):
    wrist = np.array([wrist.x, wrist.y])
    thumb_base = np.array([thumb_base.x, thumb_base.y])
    thumb_tip = np.array([thumb_tip.x, thumb_tip.y])
    vector_thumb = thumb_tip - thumb_base
    vector_wrist = wrist - thumb_base
    angle_rad = math.atan2(vector_thumb[1], vector_thumb[0]) - math.atan2(vector_wrist[1], vector_wrist[0])
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    return angle_deg

def is_finger_raised(landmarks, finger_tip_idx, finger_base_idx):
    """Checks if a specific finger is raised by comparing the tip's Y position with the base."""
    finger_tip = landmarks[finger_tip_idx]
    finger_base = landmarks[finger_base_idx]

    return finger_tip.y < finger_base.y  # Y-axis decreases as we move up

# Function to calculate wrist extension angle
def calculate_extension_angle(wrist, index_base, middle_base):
    wrist = np.array([wrist.x, wrist.y])
    index_base = np.array([index_base.x, index_base.y])
    middle_base = np.array([middle_base.x, middle_base.y])
    
    finger_base_avg = (index_base + middle_base) / 2
    vector_wrist_finger = finger_base_avg - wrist
    reference_vector = np.array([1, 0])
    
    angle_rad = math.atan2(vector_wrist_finger[1], vector_wrist_finger[0]) - math.atan2(reference_vector[1], reference_vector[0])
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    return angle_deg

def euclidean_distance(a, b):
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

# Function to calculate Euclidean distance between two landmarks
def distance(lm1, lm2):
    return np.linalg.norm(np.array([lm1.x, lm1.y]) - np.array([lm2.x, lm2.y]))


def generate_video_feed():
    global start_time ,exercise_active , max_duration , side

    cap = cv2.VideoCapture(0)  # Open the camera


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        #cv2.putText(frame, f'Exercise: {exercise}', (50, 50), 
                    #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                
          
                feedback_lines = detect_exercise(landmarks, exercise)

                if isinstance(feedback_lines, list):
                    for i, feedback in enumerate(feedback_lines):
                        cv2.putText(frame, feedback, (50, 100 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, feedback_lines, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elapsed_time = time.time() - start_time
        if exercise_active:
            cv2.putText(frame, "Good Job!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif exercise_active == False :
            cv2.putText(frame, "Keep Going :)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif elapsed_time > max_duration and exercise_active == False:
            cv2.putText(frame, "Time Exceeded!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        save_exercise_data(child_id, exercise, side ,  exercise_active)


        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as part of the HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


def detect_exercise(landmarks, exercise  ):
    #feedback = "Keep Going"
    global exercise_active  , fingers_done
    
###################################################################################
    if exercise == "finger_exercise": #yes
        feedback_lines = []  # Store feedback messages
        thumb_tip = landmarks[4]  # Thumb tip
        fingers = {
        "Index": landmarks[8],
        "Middle": landmarks[12],
        "Ring": landmarks[16],
        "Pinky": landmarks[20]}
        
        for finger, tip in fingers.items():
            dist = distance(thumb_tip, tip)  # Measure distance between thumb and fingertip

            if dist < 0.04:  # If close enough, register as pinch
                feedback_lines.append(f"{finger} Pinched!")
                if finger in fingers_done:
                    fingers_done.remove(finger)   
            else:
                feedback_lines.append(f"{finger} Released!")

        if len(fingers_done) == 0:
            exercise_active = True
            fingers_done = {'Index' , 'Middle' , 'Ring' , 'Pinky'}

    
        return feedback_lines   # Return list instead of a single string
    
################################################################################
    elif exercise == "finger_raising": #yes 
        index_raised = is_finger_raised(landmarks, 8, 5)   # Index Finger
        middle_raised = is_finger_raised(landmarks, 12, 9) # Middle Finger
        ring_raised = is_finger_raised(landmarks, 16, 13)  # Ring Finger
        pinky_raised = is_finger_raised(landmarks, 20, 17) # Pinky Finger

        raised_fingers = set()
        if index_raised:
            raised_fingers.add("Index")
        if middle_raised:
            raised_fingers.add("Middle")
        if ring_raised:
            raised_fingers.add("Ring")
        if pinky_raised:
            raised_fingers.add("Pinky")

        if raised_fingers:
            feedback = f"Raised: {', '.join(raised_fingers)}"
            if len(raised_fingers) == 4 :
                exercise_active = True
        else:
            feedback = "No fingers raised"
            exercise_active = False

###################################################################################
    elif exercise == "fist_exercise": #yes
        # Check if all fingers move towards palm
        palm = landmarks[0]
        all_fingers_closed = all(distance(landmarks[i], palm) < 0.4 for i in [4, 8, 12, 16, 20])
        
        if all_fingers_closed:
            feedback = "Full Grip Done!"
            exercise_active  = True
#################################################################################
    elif exercise == "bend_the_wrist" :
        wrist = landmarks[0]    
        middle_base = landmarks[9]
        index_base = landmarks[5]
        wrist_angle = calculate_extension_angle(wrist, index_base, middle_base)
        all_fingers_closed = all(distance(landmarks[i], wrist) < 0.2 for i in [4, 8, 12, 16, 20])

        if all_fingers_closed:
            if 290 <= wrist_angle <=320  or 260 <= wrist_angle <=300:
                feedback = "Bend"
                exercise_active = True
            else:
                feedback = "Neutral"
        else :
            feedback = "Close your hand correctly"
    
#############################################################################
    elif exercise == "finger_adduction":
        index_tip, middle_tip, ring_tip, pinky_tip = landmarks[6], landmarks[10], landmarks[14], landmarks[19]

        index_middle_dist = distance(index_tip, middle_tip)
        middle_ring_dist = distance(middle_tip, ring_tip)
        ring_pinky_dist = distance(ring_tip, pinky_tip)

        if index_middle_dist < 0.1 and middle_ring_dist < 0.1 and ring_pinky_dist < 0.1:
            feedback = "Fingers Adducted (Closed Together)"
        else:
            feedback = "Fingers Abducted (Spread Apart)"
            exercise_active = True
#############################################################################


    elif exercise == "opening_and_closing":
        wrist = landmarks[0]
        fingertips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
    # Calculate the distance between wrist and each fingertip
        distances = [distance(tip, wrist) for tip in fingertips]

    # If all distances are greater than threshold_open, hand is open
        if all(dist > threshold_open for dist in distances):
            feedback = "Hand Open"
    # If all distances are smaller than threshold_closed, hand is closed (fist)
        elif all(dist < threshold_closed for dist in distances):
            feedback = "Fist Closed"
            exercise_active = True
        else:
            feedback =  "Hand in Transition"
            
#############################################################################

    elif exercise == "wrist_rotation" : #yes
        wrist = landmarks[0]
        thumb_base = landmarks[2]
        thumb_tip = landmarks[4]
        palm = landmarks[0]
        thumb_angle = calculate_360_angle(wrist, thumb_base, thumb_tip)
    
        if 160 <= thumb_angle <= 200:
            feedback = "Half_Rotate"
        elif 120 <= thumb_angle <= 150 or 210 <= thumb_angle <= 220:
            feedback = "Full_Rotete"
            exercise_active = True
################################################################################
    elif exercise == "bending_exercise": #yes
        wrist = landmarks[0]
        thumb_base = landmarks[2]
        thumb_tip = landmarks[4]
        palm = landmarks[0]
        all_fingers_closed = all(distance(landmarks[i], palm) < 0.2 for i in [4, 8, 12, 16, 20])
        thumb_angle = calculate_360_angle(wrist, thumb_base, thumb_tip)

        if all_fingers_closed:      
            if 160 <= thumb_angle <= 200:
                feedback = "UP"
            elif 120 <= thumb_angle <= 150 or 210 <= thumb_angle <= 220:
                feedback = "DOWN"
        else :
            feedback = "Close your hand correctly"
        
            exercise_active = True

    return feedback
