from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import time
import logging
import os

# Import your models
import body_model_2
import arm_model_home
import hand_model_home
import hand_model_center

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

# Default port handling
PORT = int(os.getenv("PORT", 5000))  # Ensure PORT is an integer

@app.route('/start_exercise/<mode>/<child_id>/<exercise_name>', methods=['GET'])
def start_exercise(mode, child_id, exercise_name):
    """
    Route to start an exercise based on the mode, child ID, and exercise name.
    """
    logging.debug(f"Request received: mode={mode}, child_id={child_id}, exercise_name={exercise_name}")
    try:
        if mode == 'body':
            return generate_response(body_model_2, child_id, exercise_name)
        elif mode == 'arm':
            return generate_response(arm_model_home, child_id, exercise_name, request.args.get('side'))
        elif mode == 'hand':
            return generate_response(hand_model_home, child_id, exercise_name, request.args.get('side'))
        elif mode == 'center':
            return generate_response(hand_model_center, None, exercise_name)
        else:
            logging.warning(f"Invalid mode: {mode}")
            return jsonify({"error": "Invalid mode"}), 400
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({"error": "Internal server error"}), 500

def generate_response(model, child_id, exercise_name, side=None):
    """
    Helper function to initialize the model and generate a video feed response.
    """
    initialize_model(model, child_id, exercise_name, side)
    return Response(model.generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

def initialize_model(model, child_id, exercise_name, side=None):
    """
    Initializes the model with common attributes.
    """
    model.exercise = exercise_name
    model.start_time = time.time()
    if hasattr(model, 'child_id') and child_id:
        model.child_id = child_id
    if hasattr(model, 'side') and side:
        model.side = side
    if hasattr(model, 'exercise_active'):
        model.exercise_active = False
    if hasattr(model, 'max_duration'):
        model.max_duration = 120
    if hasattr(model, 'distance_threshold_close') and hasattr(model, 'distance_threshold_far'):
        thresholds = model.exercises.get(exercise_name, {})
        model.distance_threshold_close = thresholds.get('distance_threshold_close')
        model.distance_threshold_far = thresholds.get('distance_threshold_far')

if __name__ == "__main__":
    # For development only. Use a WSGI server like Gunicorn for production.
    app.run(host='0.0.0.0', port=PORT)


