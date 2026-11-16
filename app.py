from flask import Flask, render_template, Response, jsonify, request, abort
import cv2
import pyttsx3
from cvzone.HandTrackingModule import HandDetector
from collections import Counter
import csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import speech_recognition as sr
import threading
from motion_recognition import MotionRecognizer
from automatic_entry import AutomaticEntry, REST_LABEL
from hand_features import isl_features, isl_motion_features
from training_data import delete_label_samples, fit_model, load_model_state, save_model_state
from tensorflow_motion import (MAX_RECORDING_FRAMES, NO_MOTION, TensorFlowMotionRecognizer,fit_motion, frame_features, has_motion, load_motion_state, prepare_sequence,save_motion_state,)
app = Flask(__name__)
MODEL_PATH = "models/sign_language_model.pkl"
DATA_PATH = "sign_data.csv"
MOTION_PATH = "models/motion_data.pkl"
MODE_PATHS = {
    "asl": ("models/sign_language_model.pkl", "sign_data.csv",
            "models/motion_data.pkl", "signpictures"),
    "isl": ("models/isl_sign_language_model.pkl", "isl_sign_data.csv",
            "models/isl_motion_data.pkl", "signpictures_isl"),
    "pretrained_asl": ("models/pretrained_asl_model.pkl", None,
                       "models/pretrained_asl_motion.pkl", "signpictures_pretrained_asl"),
    "pretrained_isl": ("models/pretrained_isl_model.pkl", None,
                       "models/pretrained_isl_motion.pkl", "signpictures_pretrained_isl"),
}
active_language = "asl"
mode_generation = 0
model, disabled_motion = load_model_state(MODEL_PATH)
motion_state = load_motion_state(MOTION_PATH)
tensorflow_motion = TensorFlowMotionRecognizer(motion_state)
state_lock = threading.RLock()
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
current_prediction = ""
sentence = ""
predictions = []
STABLE_FRAMES = 10
motion_recognizer = MotionRecognizer()
dynamic_prediction_frames = 0
automatic_entry = AutomaticEntry()
is_training = False
training_label = ""
training_count = 0
training_motion = False
training_motion_frames = []
recognized_text = ""
is_listening = False
def speak_text(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        print(f"Speaking: {text}")
        engine.say(text)
        engine.runAndWait()
        print("Speech complete")
    except Exception as e:
        print(f"Text-to-speech error: {e}")
def listen_to_speech():
    global recognized_text, is_listening
    try:
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 4000
        mic = sr.Microphone()
        print("Initializing microphone...")
        with mic as source:
            print("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Listening for speech...")
            print(f"Energy threshold: {recognizer.energy_threshold}")
            audio = recognizer.listen(source, timeout=20, phrase_time_limit=20)
        print("Processing audio...")
        recognized_text = recognizer.recognize_google(audio, language='en-US')
        print(f"Recognized: {recognized_text}")
    except sr.UnknownAudioException as e:
        print(f"UnknownAudioException: {e}")
        recognized_text = "Could not understand speech. Speak clearer and louder."
    except sr.RequestError as e:
        print(f"RequestError: {e}")
        recognized_text = f"API Error: Check internet connection"
    except sr.Timeout as e:
        print(f"Timeout: {e}")
        recognized_text = "No speech detected. Try speaking louder or closer."
    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")
        recognized_text = f"Error: {str(e)}"
    finally:
        is_listening = False
def extract_hand_features(landmarks):
    """Extract normalized hand landmarks"""
    if not landmarks:
        return None
    row = []
    x0 = landmarks[0][0]
    y0 = landmarks[0][1]
    for lm in landmarks:
        row.append(lm[0] - x0)
        row.append(lm[1] - y0)
    return row
def get_sign_image_path(letter):
    """Get path to sign language image"""
    letter = letter.upper()
    signpics_dir = "signpictures"
    for ext in ['.jpg', '.png', '.jpeg', '.gif']:
        path = os.path.join(signpics_dir, f"{letter}{ext}")
        if os.path.exists(path):
            return f"/sign_image/{letter}"
    return None
def expected_feature_count(language):
    return 85 if language in ("isl", "pretrained_isl") else 42
def static_model_is_compatible():
    if model is None:
        return False
    feature_count = getattr(model, "n_features_in_", None)
    if feature_count is None:
        return not active_language.startswith("pretrained_")
    return feature_count == expected_feature_count(active_language)
def get_supported_vocabulary():
    labels = set()
    if static_model_is_compatible():
        labels.update(str(label) for label in model.classes_)
    if tensorflow_motion.network is not None:
        labels.update(str(label) for label in tensorflow_motion.labels if label != NO_MOTION)
    return sorted(("Rest position" if label == REST_LABEL else label for label in labels),key=str.casefold,)
def generate_frames(translate=False, generation=None):
    global current_prediction
    global predictions
    global is_training
    global training_label
    global training_count
    global dynamic_prediction_frames
    global sentence
    while True:
        with state_lock:
            if generation is not None and generation != mode_generation:
                return
            success, img = cap.read()
            if not success:
                continue
            hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            landmarks = hand["lmList"]
            with state_lock:
                if generation is not None and generation != mode_generation:
                    return
                is_isl = active_language in ("isl", "pretrained_isl")
                features = (isl_features(hands) if is_isl
                            else extract_hand_features(landmarks))
                motion_input = hands if is_isl else landmarks
                if is_training:
                    motion_recognizer.reset()
                    tensorflow_motion.reset()
                    dynamic_prediction_frames = 0
                    if training_motion:
                        if len(training_motion_frames) < MAX_RECORDING_FRAMES:
                            training_motion_frames.append(
                                isl_motion_features(hands) if is_isl
                                else frame_features(landmarks))
                        training_count = len(training_motion_frames)
                    elif features:
                        with open(DATA_PATH, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(features + [training_label])
                        training_count += 1
                    display_text = f"Training {training_label}: {training_count}"
                else:
                    learned_motion = set(motion_state["labels"]) - {NO_MOTION}
                    static_prediction = None
                    if features and static_model_is_compatible():
                        static_prediction = str(model.predict([features])[0])
                        if static_prediction in learned_motion:
                            static_prediction = None
                    motion_sign = tensorflow_motion.update(motion_input)
                    if not motion_sign and active_language == "asl":
                        motion_sign = motion_recognizer.update(landmarks, disabled_motion | learned_motion)
                    if motion_sign:
                        current_prediction = motion_sign
                        predictions.clear()
                        dynamic_prediction_frames = STABLE_FRAMES * 3
                    elif dynamic_prediction_frames > 0:
                        dynamic_prediction_frames -= 1
                    elif static_prediction is not None:
                        predictions.append(static_prediction)
                        if len(predictions) > STABLE_FRAMES:
                            predictions.pop(0)
                        if predictions:
                            current_prediction = Counter(predictions).most_common(1)[0][0]
                    elif features and model is not None:
                        current_prediction = ""
                        predictions.clear()
                    if translate:
                        entry_label = motion_sign
                        if not entry_label:
                            if dynamic_prediction_frames > 0 and static_prediction != REST_LABEL:
                                entry_label = None
                            else:
                                entry_label = static_prediction
                        entry = automatic_entry.observe(entry_label, motion=bool(motion_sign))
                        if entry:
                            sentence += entry
                    display_sign = "Rest position" if current_prediction == REST_LABEL else current_prediction
                    display_text = f"Detected: {display_sign}"
                cv2.putText(img, display_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
        else:
            with state_lock:
                motion_recognizer.reset()
                tensorflow_motion.reset()
                if is_training and training_motion:
                    training_motion_frames.clear()
                    training_count = 0
                if translate:
                    automatic_entry.reset()
                    current_prediction = ""
                    predictions.clear()
                    dynamic_prediction_frames = 0
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' +frame +b'\r\n')
@app.route('/')
def index():
    return render_template('choose_language.html', pretrained=False)
@app.route('/pretrained')
def pretrained():
    return render_template('choose_language.html', pretrained=True)
@app.route('/app/<language>')
def open_app(language):
    global active_language, mode_generation, MODEL_PATH, DATA_PATH, MOTION_PATH
    global model, disabled_motion, motion_state, tensorflow_motion
    global current_prediction, sentence, dynamic_prediction_frames
    if language not in MODE_PATHS:
        abort(404)
    with state_lock:
        if language != active_language:
            if is_training:
                return "Stop recording before switching sign language.", 409
            new_model_path, new_data_path, new_motion_path, _ = MODE_PATHS[language]
            new_model, new_disabled = (load_model_state(new_model_path) if os.path.exists(new_model_path)else (None, set()))
            new_motion_state = load_motion_state(new_motion_path)
            extractor = (isl_motion_features if language in ("isl", "pretrained_isl")
                         else frame_features)
            new_tensorflow_motion = TensorFlowMotionRecognizer(new_motion_state, extractor)
            active_language = language
            MODEL_PATH, DATA_PATH, MOTION_PATH = new_model_path, new_data_path, new_motion_path
            model, disabled_motion = new_model, new_disabled
            motion_state, tensorflow_motion = new_motion_state, new_tensorflow_motion
            sentence = ""
            current_prediction = ""
            dynamic_prediction_frames = 0
            predictions.clear()
            motion_recognizer.reset()
            automatic_entry.reset()
        mode_generation += 1
        pretrained_mode = language.startswith("pretrained_")
        if pretrained_mode and model is not None and not static_model_is_compatible():
            app.logger.error(
                "Pre-trained %s model expects %s features; this mode provides %s.",
                "ISL" if language.endswith("_isl") else "ASL",
                getattr(model, "n_features_in_", "an unknown number of"),
                expected_feature_count(language),
            )
            model_status = (
                "Model input mismatch: this mode requires "
                f"{expected_feature_count(language)} features; "
                "install a model trained with this mode's preprocessing."
            )
        elif pretrained_mode and model is None and tensorflow_motion.network is None:
            dataset_name = "isl_sign_data.csv" if language == "pretrained_isl" else "sign_data.csv"
            model_status = (
                f"Model unavailable. Add the labeled dataset {dataset_name} to the project root "
                f"and a compatible model at models/{language}_model.pkl. "
                f"Model labels define its vocabulary; expected inputs are "
                f"{expected_feature_count(language)} features per sample."
            )
        elif pretrained_mode and model is None:
            model_status = "Static model not installed; motion recognition is available."
        elif pretrained_mode:
            model_status = "Pre-trained model loaded."
        else:
            model_status = ""
    return render_template(
        'index.html', language=language,
        display_language="ISL" if language in ("isl", "pretrained_isl") else "ASL",
            pretrained_mode=pretrained_mode,
            model_status=model_status,
            supported_vocabulary=get_supported_vocabulary() if pretrained_mode else [],
    )
@app.before_request
def ensure_active_language():
    language = request.args.get("language")
    if language and language != active_language:
        return jsonify({"status": "error", "message": "Sign language changed. Reload this page."}), 409
    if (active_language.startswith("pretrained_") and request.endpoint in {"start_training", "stop_training", "train_model_route","delete_training_data_route", "get_training_count",}):
        return jsonify({"status": "error", "message": "Training is unavailable in pre-trained mode."}), 403
@app.route('/video_feed')
def video_feed():
    translate = request.args.get('mode') == 'translate'
    with state_lock:
        generation = mode_generation
        if translate:
            automatic_entry.reset()
    return Response(
        generate_frames(translate=translate, generation=generation),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/sign_image/<letter>')
def sign_image(letter):
    """Serve sign language images"""
    letter = letter.upper()
    signpics_dir = MODE_PATHS[active_language][3]
    for ext in ['.jpg', '.png', '.jpeg', '.gif']:
        path = os.path.join(signpics_dir, f"{letter}{ext}")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return f.read(), 200, {'Content-Type': 'image/jpeg'}
    return "", 404
@app.route('/current_sign')
def current_sign():
    with state_lock:
        rest_trained = static_model_is_compatible() and REST_LABEL in model.classes_
        sign = "Rest position" if current_prediction == REST_LABEL else current_prediction
        return jsonify({"sign": sign, "rest_trained": bool(rest_trained),"model_available": static_model_is_compatible() or tensorflow_motion.network is not None,"ready": automatic_entry.ready})
@app.route('/add_letter')
def add_letter():
    global sentence
    with state_lock:
        if current_prediction and current_prediction != REST_LABEL:
            sentence += current_prediction
        return jsonify({"sentence": sentence})
@app.route('/space')
def add_space():
    global sentence
    with state_lock:
        sentence += " "
        return jsonify({"sentence": sentence})
@app.route('/backspace')
def backspace():
    global sentence
    with state_lock:
        sentence = sentence[:-1]
        return jsonify({"sentence": sentence})
@app.route('/clear')
def clear():
    global sentence
    with state_lock:
        sentence = ""
        return jsonify({"sentence": sentence})
@app.route('/speak')
def speak():
    if sentence:
        thread = threading.Thread(target=speak_text, args=(sentence,))
        thread.daemon = True
        thread.start()
        return jsonify({"status": "speaking", "sentence": sentence})
    return jsonify({"status": "nothing to speak", "sentence": sentence})
@app.route('/sentence')
def get_sentence():
    return jsonify({"sentence": sentence})
@app.route('/start_training', methods=['POST'])
def start_training():
    global is_training, training_label, training_count, training_motion
    data = request.get_json(silent=True) or {}
    label = data.get('label', 'A')
    motion = data.get('motion', False)
    if not isinstance(label, str) or not label.strip() or not isinstance(motion, bool):
        return jsonify({"status": "error", "message": "Provide a label and a valid motion setting."}), 400
    if label.strip() == NO_MOTION:
        return jsonify({"status": "error", "message": "That label is reserved."}), 400
    if label.strip().casefold() == REST_LABEL.casefold():
        if motion:
            return jsonify({"status": "error", "message": "Rest position must be recorded without motion."}), 400
        label = REST_LABEL
    with state_lock:
        if is_training:
            return jsonify({"status": "error", "message": "Stop recording first."}), 409
        training_label = label.strip()
        if len(training_label) == 1:
            training_label = training_label.upper()
        training_motion = motion
        training_motion_frames.clear()
        automatic_entry.reset()
        is_training = True
        training_count = 0
        return jsonify({"status": "training started", "label": training_label})
@app.route('/stop_training', methods=['POST'])
def stop_training():
    global is_training, training_count, motion_state
    with state_lock:
        if not is_training:
            return jsonify({"status": "error", "message": "Not recording."}), 409
        is_training = False
        count = training_count
        training_count = 0
        if training_motion:
            try:
                sequence = prepare_sequence(training_motion_frames)
                if not has_motion(sequence):
                    raise ValueError("No motion detected. Record the full movement again.")
                recordings = motion_state["recordings"] + [{"label": training_label, "frames": list(training_motion_frames)}]
                updated = {**motion_state, "recordings": recordings}
                save_motion_state(MOTION_PATH, updated)
                motion_state = updated
                training_motion_frames.clear()
                return jsonify({"status": "training stopped", "samples": 1,"motion": True, "frames": count})
            except (ValueError, OSError) as e:
                app.logger.warning("Could not save motion recording: %s", e)
                return jsonify({"status": "error", "message": str(e)}), 400
        return jsonify({"status": "training stopped", "samples": count})
@app.route('/delete_training_data', methods=['POST'])
def delete_training_data_route():
    global model, disabled_motion, current_prediction, dynamic_prediction_frames
    global motion_state, tensorflow_motion
    data = request.get_json(silent=True) or {}
    label = data.get('label')
    if not isinstance(label, str) or not label.strip():
        return jsonify({"status": "error","message": "Select a valid character to delete."}), 400
    with state_lock:
        if is_training:
            return jsonify({"status": "error","message": "Stop recording before deleting training data."}), 409
        try:
            try:
                samples = pd.read_csv(DATA_PATH, header=None)
            except (pd.errors.EmptyDataError, FileNotFoundError):
                samples = pd.DataFrame()
            normalized = label.strip().casefold()
            if samples.empty:
                remaining = samples
            else:
                remaining = samples[
                    samples.iloc[:, -1].astype(str).str.strip().str.casefold() != normalized
                ]
            rebuilt_model = fit_model(remaining)
            next_disabled = disabled_motion.copy()
            if normalized in ("j", "z"):
                next_disabled.add(normalized.upper())

            remaining_recordings = [
                recording for recording in motion_state["recordings"]
                if recording["label"].strip().casefold() != normalized
            ]
            removed_recordings = len(motion_state["recordings"]) - len(remaining_recordings)
            if removed_recordings:
                trained_labels = {
                    name for name in motion_state["labels"]
                    if name != NO_MOTION and name.strip().casefold() != normalized
                }
                trained_recordings = [
                    recording for recording in remaining_recordings
                    if recording["label"] in trained_labels
                ]
                next_motion = fit_motion(trained_recordings)
                next_motion["recordings"] = remaining_recordings
                extractor = isl_motion_features if active_language == "isl" else frame_features
                next_recognizer = TensorFlowMotionRecognizer(next_motion, extractor)
            else:
                next_motion = motion_state
                next_recognizer = tensorflow_motion
            save_model_state(MODEL_PATH, rebuilt_model, next_disabled)
            model = rebuilt_model
            disabled_motion = next_disabled
            save_motion_state(MOTION_PATH, next_motion)
            motion_state = next_motion
            tensorflow_motion = next_recognizer
            current_prediction = ""
            predictions.clear()
            dynamic_prediction_frames = 0
            motion_recognizer.reset()
            automatic_entry.reset()
            removed = delete_label_samples(DATA_PATH, label) if os.path.exists(DATA_PATH) else 0
            return jsonify({
                "status": "success",
                "label": label.strip(),
                "removed": removed,
                "removed_motion": removed_recordings
            })
        except FileNotFoundError:
            return jsonify({"status": "error","message": "Training data file was not found."}), 404
        except (OSError, ValueError, pd.errors.ParserError) as e:
            app.logger.exception("Could not forget training label")
            return jsonify({"status": "error", "message": str(e)}), 500
@app.route('/train_model', methods=['POST'])
def train_model_route():
    global model, disabled_motion, current_prediction, dynamic_prediction_frames
    global motion_state, tensorflow_motion
    with state_lock:
        if is_training:
            return jsonify({"status": "error", "message": "Stop recording before training."}), 409
        try:
            try:
                data = pd.read_csv(DATA_PATH, header=None)
            except (pd.errors.EmptyDataError, FileNotFoundError):
                data = pd.DataFrame()
            if data.empty and not motion_state["recordings"]:
                return jsonify({"status": "error", "message": "No training data. Collect data first!"})
            accuracy = "N/A"
            labels = set()
            model_new = None
            if not data.empty:
                X = data.iloc[:, :-1]
                y = data.iloc[:, -1]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                evaluation_model = fit_model(pd.concat([X_train, y_train], axis=1))
                accuracy = f"{accuracy_score(y_test, evaluation_model.predict(X_test)):.2%}"
                model_new = fit_model(data)
                labels = {str(label).strip().upper() for label in y}
            next_motion = fit_motion(motion_state["recordings"])
            extractor = isl_motion_features if active_language == "isl" else frame_features
            next_recognizer = TensorFlowMotionRecognizer(next_motion, extractor)
            next_disabled = disabled_motion - labels
            save_model_state(MODEL_PATH, model_new, next_disabled)
            save_motion_state(MOTION_PATH, next_motion)
            model = model_new
            disabled_motion = next_disabled
            motion_state = next_motion
            tensorflow_motion = next_recognizer
            current_prediction = ""
            predictions.clear()
            dynamic_prediction_frames = 0
            motion_recognizer.reset()
            automatic_entry.reset()
            return jsonify({"status": "success","accuracy": accuracy,"samples": len(data),"motion_recordings": len(next_motion["recordings"])})
        except Exception as e:
            app.logger.exception("Could not train model")
            return jsonify({"status": "error", "message": str(e)})
@app.route('/training_count')
def get_training_count():
    return jsonify({"count": training_count})
@app.route('/start_listening', methods=['POST'])
def start_listening():
    global is_listening, recognized_text
    if is_listening:
        return jsonify({"status": "already listening"})
    recognized_text = ""
    is_listening = True
    thread = threading.Thread(target=listen_to_speech)
    thread.daemon = True
    thread.start()
    return jsonify({"status": "listening started"})
@app.route('/stop_listening', methods=['POST'])
def stop_listening():
    global is_listening
    is_listening = False
    return jsonify({"status": "listening stopped", "text": recognized_text})
@app.route('/get_recognized_text')
def get_recognized_text():
    return jsonify({"text": recognized_text, "is_listening": is_listening})
@app.route('/test_speak', methods=['POST'])
def test_speak():
    try:
        data = request.json
        text = data.get('text', 'Hello, this is a test')
        thread = threading.Thread(target=speak_text, args=(text,))
        thread.daemon = True
        thread.start()
        return jsonify({"status": "speaking", "text": text})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
def test_microphone():
    try:
        mic = sr.Microphone()
        with mic as source:
            recognizer = sr.Recognizer()
            print("Testing microphone - speak now...")
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return jsonify({"status": "success", "message": "Microphone works! Heard: " + text})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Microphone error: {str(e)}"})
@app.route('/text_to_signs', methods=['POST'])
def text_to_signs():
    data = request.json
    text = data.get('text', '')
    signs = [c.upper() for c in text if c.isalpha()]
    return jsonify({"signs": signs, "text": text})
if __name__ == '__main__':
    app.run(debug=False)