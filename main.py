from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import time
import logging

# Import your models
import body_model_2
import arm_model_home
import hand_model_home
import hand_model_center

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

@app.route('/start_exercise/<mode>/<child_id>/<exercise_name>', methods=['GET'])
def start_exercise(mode, child_id, exercise_name):
    logging.debug(f"Request received: mode={mode}, child_id={child_id}, exercise_name={exercise_name}")
    try:
        if mode == 'body':
            return Response(start_body_v1(child_id, exercise_name), mimetype='multipart/x-mixed-replace; boundary=frame')
        elif mode == 'arm':
            return Response(start_arm(child_id, exercise_name, request.args.get('side', None)), mimetype='multipart/x-mixed-replace; boundary=frame')
        elif mode == 'hand':
            return Response(start_hand(child_id, exercise_name, request.args.get('side', None)), mimetype='multipart/x-mixed-replace; boundary=frame')
        elif mode == 'center':
            return Response(start_center(exercise_name), mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            return jsonify({"error": "Invalid mode"}), 400
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({"error": "Internal server error"}), 500


def start_body_v1(child_id, exercise_name):
    body_model_2.exercise = exercise_name
    body_model_2.child_id = child_id
    body_model_2.start_time = time.time()
    body_model_2.distance_threshold_close, body_model_2.distance_threshold_far = body_model_2.exercises[exercise_name].values()
    
    return body_model_2.generate_video_feed()


def start_arm(child_id, exercise_name , side=None):
    arm_model_home.exercise = exercise_name
    arm_model_home.child_id = child_id
    arm_model_home.start_time = time.time()
    if side:
        arm_model_home.side = side  # Only set if side param is provided
    return arm_model_home.generate_video_feed()


def start_hand(child_id , exercise_name , side = None):
    hand_model_home.exercise = exercise_name
    hand_model_home.child_id = child_id
    hand_model_home.start_time = time.time()
    hand_model_home.exercise_active = False
    hand_model_home.max_duration = 120
    return hand_model_home.generate_video_feed()

def start_center(exercise_name):
    hand_model_center.exercise = exercise_name
    hand_model_center.start_time = time.time()  # Uncommented and initialized
    hand_model_center.exercise_active = False
    hand_model_center.max_duration = 120  # Uncommented and initialized
    return hand_model_center.generate_video_feed()


if __name__ == '__main__':
    # Test start_body_v1
    print(start_body_v1("123", "exercise1"))
    # Test start_arm
    print(start_arm("123", "exercise1", "left"))
    # Test start_hand
    print(start_hand("123", "exercise1", "right"))
    # Test start_center
    print(start_center("exercise1"))
