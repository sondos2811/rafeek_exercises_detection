import cv2
import mediapipe as mp
import numpy as np
import time
import math
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import threading

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

exercise_active = False  # Track if exercise is being performed
start_time = time.time()
max_duration = 120 
fingers_done = {'Index' , 'Middle' , 'Ring' , 'Pinky'}
exercise = None  # Initialize the exercise variable

# Global variable to track the video processing thread
video_thread = None
video_thread_running = False  # To prevent multiple threads from starting


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

# Function to calculate Euclidean distance between two landmarks
def distance(lm1, lm2):
    return np.linalg.norm(np.array([lm1.x, lm1.y]) - np.array([lm2.x, lm2.y]))


def generate_video_feed():
    global start_time ,exercise_active , max_duration

    cap = cv2.VideoCapture(0)  # Open the camera

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                
          
                feedback_lines = detect_exercise(landmarks, exercise)

                if isinstance(feedback_lines, list):
                    for i, feedback in enumerate(feedback_lines):
                        cv2.putText(frame, feedback, (50, 100 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, feedback_lines, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elapsed_time = time.time() - start_time
        if exercise_active:
            cv2.putText(frame, "Good Job!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        elif exercise_active == False :
            cv2.putText(frame, "Keep Going :)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif elapsed_time > max_duration and exercise_active == False:
            cv2.putText(frame, "Time Exceeded!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as part of the HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def detect_exercise(landmarks, exercise  ):
    feedback = ""
    global exercise_active , fingers_done
    
###################################################################################
    if exercise == "finger_opposition": #DONE
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


    elif exercise == "thumb-index_press":  #DONE
        exercise_active = False
        thumb_tip = landmarks[4]
        thumb_base = landmarks[2]
        index_tip = landmarks[8]
        index_base = landmarks[5]

        fingers = {
            "Middle": (landmarks[12], landmarks[9]),
            "Ring": (landmarks[16], landmarks[13]),
            "Pinky": (landmarks[20], landmarks[17])
        }

        # Condition 1: Ensure all fingers and thumb are straight (tips far from base)
        all_fingers_straight = all(distance(tip, base) > 0.05 for tip, base in fingers.values()) and distance(thumb_tip, thumb_base) > 0.05

        # Condition 2: Check if thumb and index finger are pressing (close to each other)
        pressing = distance(thumb_tip, index_tip) < 0.04  # Adjust threshold if needed

        if all_fingers_straight and pressing:
            feedback = "Thumb-Index Press Done Correctly!"
            exercise_active = True
        elif not all_fingers_straight:
            feedback = "Keep All Fingers and Thumb Straight!"
        elif not pressing:
            feedback = "Press the Putty Between Thumb and Index Finger!"

###################################################################################

    elif exercise == "thumb_press": # DONE
        exercise_active = False
        thumb_tip = landmarks[4]
        pinky_base = landmarks[17]
        dist = distance(thumb_tip, pinky_base)
        
        if dist < 0.05:  # Detect pressing motion
            feedback = "Thumb Pressed Correctly!"
            exercise_active  = True

#############################################################################

    elif exercise == "thumb_extend": # Done
        exercise_active = False
        thumb_tip = landmarks[4]
        thumb_base = landmarks[13]

        fingers = {
            "Index": (landmarks[8], landmarks[5]),
            "Middle": (landmarks[12], landmarks[9]),
            "Ring": (landmarks[16], landmarks[13]),
            "Pinky": (landmarks[20], landmarks[17])
        }

        # Condition 1: All fingers should be extended (tips far from base)
        all_fingers_extended = all(distance(tip, base) > 0.09 for tip, base in fingers.values())

        # Condition 2: Thumb should be bent (holding position)
        thumb_bent = distance(thumb_tip, thumb_base) < 0.04

        if all_fingers_extended and thumb_bent:
            feedback = "Thumb Holding the Ball Correctly!"
            exercise_active  = True
        elif not all_fingers_extended:
            feedback = "Keep All Fingers Extended!"
        elif not thumb_bent:
            feedback = "Bend Your Thumb to Hold the Ball!"

#############################################################################

    elif exercise == "full_grip": #DONE
        exercise_active = False
        # Check if all fingers move towards palm
        palm = landmarks[0]
        all_fingers_closed = all(distance(landmarks[i], palm) < 0.2 for i in [4, 8, 12, 16, 20])
        
        if all_fingers_closed:
            feedback = "Full Grip Done!"
            exercise_active  = True
    
#############################################################################

    elif exercise == "ball_grip": #DONE
        exercise_active = False
        # Check if all fingers move towards palm
        palm = landmarks[0]
        all_fingers_closed = all(distance(landmarks[i], palm) < 0.3 for i in [4, 8, 12, 16, 20])
        
        if all_fingers_closed:
            feedback = "ball Grip Done!"
            exercise_active  = True

#############################################################################

    elif exercise == "finger_hook":  #DONE
        exercise_active = False
        fingers = {
            "Index": (landmarks[8], landmarks[6], landmarks[5]),
            "Middle": (landmarks[12], landmarks[10], landmarks[9]),
            "Ring": (landmarks[16], landmarks[14], landmarks[13]),
            "Pinky": (landmarks[20], landmarks[18], landmarks[17])
        }

        hook_grip_detected = True  # Assume correct, then validate
        
        for finger, (tip, middle, base) in fingers.items():
            # Condition 1: Fingertip must be closer to base than its normal extended position
            if distance(tip, base) > 0.08:  # Adjust threshold based on testing
                hook_grip_detected = False
                feedback = f"{finger} is not curled enough!"
                break  # Stop checking if one finger is incorrect
            
            # Condition 2: Middle joint must also move toward the base
            if distance(middle, base) < distance(tip, base):  
                hook_grip_detected = False
                feedback = f"{finger} is not bending properly!"
                break

        if hook_grip_detected:
            feedback = "Finger Hook Done Correctly!"
            exercise_active  = True

####################################################################

    elif exercise == "finger_pinch":  #DONE
        exercise_active = False
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        wrist = landmarks[0]  # Wrist landmark
        
        other_fingers = {
            "Middle": landmarks[12],
            "Ring": landmarks[16],
            "Pinky": landmarks[20]
        }

        # Condition 1: Thumb and Index are close together (holding the ball)
        holding_ball = distance(thumb_tip, index_tip) < 0.2  # Adjust threshold if needed

        # Condition 2: Other fingers are bent close to the wrist
        fingers_near_wrist = all(distance(finger, wrist) < 0.1 for finger in other_fingers.values())

        if holding_ball and fingers_near_wrist:
            feedback = "Holding Ball Correctly with Fingers Near Wrist!"
            exercise_active  = True
        elif not holding_ball:
            feedback = "Ensure Thumb and Index Grip the Ball!"
        elif not fingers_near_wrist:
            feedback = "Curl Other Fingers Closer to the Wrist!"

#########################################################################
    elif exercise == "pinch":  #DONE
        exercise_active = False
        feedback_lines = []  # Store feedback messages
    
        thumb_tip = landmarks[4]
        fingers = {
        "Index": landmarks[8],
        "Middle": landmarks[12],
        "Ring": landmarks[16],
        "Pinky": landmarks[20]
        }

    # Check if each finger is close to the thumb
        all_pinch_correct = True
        for finger_name, finger_tip in fingers.items():
            if distance(finger_tip, thumb_tip) > 0.2:  # Adjust threshold based on testing
                feedback_lines.append(f"Bring {finger_name} Closer to Thumb!")
                all_pinch_correct = False

        if all_pinch_correct:
            feedback_lines.append("Full Finger Pinch Done Correctly!")
            exercise_active  = True

        return feedback_lines  # Return list of feedback messages


################################################################################
    elif exercise == "wrist_Extension" :
        exercise_active = False
        wrist = landmarks[0]    
        middle_base = landmarks[9]
        index_base = landmarks[5]
        wrist_angle = calculate_extension_angle(wrist, index_base, middle_base)
                
        if 290 <= wrist_angle <=320  or 260 <= wrist_angle <=300:
            feedback = "Extended"
            exercise_active = True
        else:
            feedback = "Neutral"

    elif exercise == "wrist_curl":
        wrist = landmarks[0]
        thumb_base = landmarks[2]
        thumb_tip = landmarks[4]
        thumb_angle = calculate_360_angle(wrist, thumb_base, thumb_tip)
                
        if 160 <= thumb_angle <= 200:
            feedback = "UP"
        elif 120 <= thumb_angle <= 150 or 210 <= thumb_angle <= 220:
            feedback = "DOWN"
            exercise_active = True

    return feedback


