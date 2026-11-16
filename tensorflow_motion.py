import os
import tempfile
from collections import Counter, deque

import joblib
import numpy as np


FRAMES = 30
MIN_RECORDINGS = 3
MAX_RECORDING_FRAMES = 300
NO_MOTION = "__no_motion__"


def frame_features(landmarks):
    wrist = landmarks[0]
    return [
        value
        for landmark in landmarks
        for value in (landmark[0] - wrist[0], landmark[1] - wrist[1])
    ] + [wrist[0], wrist[1]]


def prepare_sequence(frames):
    data = np.asarray(frames, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] not in (44, 87) or len(data) < 10:
        raise ValueError("Record at least 10 frames with your hand visible.")
    scale = np.linalg.norm(data[:, 10:12] - data[:, 34:36], axis=1).mean()
    if scale < 1:
        raise ValueError("Could not measure the hand. Try recording again.")
    data = data.copy()
    data[:, -2:] -= data[0, -2:]
    data /= scale
    old_times = np.linspace(0, 1, len(data))
    new_times = np.linspace(0, 1, FRAMES)
    return np.stack(
        [np.interp(new_times, old_times, data[:, i]) for i in range(data.shape[1])],
        axis=1,
    ).astype(np.float32)


def has_motion(sequence):
    return np.max(np.linalg.norm(sequence - sequence[0], axis=1)) >= 0.15


def load_motion_state(path):
    if not os.path.exists(path):
        return {"recordings": [], "labels": [], "weights": None}
    return joblib.load(path)


def save_motion_state(path, state):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=os.path.dirname(os.path.abspath(path)), delete=False
        ) as file:
            temporary_path = file.name
        joblib.dump(state, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def build_network(label_count, feature_count=44):
    import tensorflow as tf

    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(FRAMES, feature_count)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(label_count, activation="softmax"),
    ])


def fit_motion(recordings):
    if not recordings:
        return {"recordings": [], "labels": [], "weights": None}
    counts = Counter(record["label"] for record in recordings)
    insufficient = [label for label, count in counts.items() if count < MIN_RECORDINGS]
    if insufficient:
        raise ValueError(
            f"Record at least {MIN_RECORDINGS} separate movements for: "
            + ", ".join(sorted(insufficient))
        )
    labels = sorted(counts) + [NO_MOTION]
    sequences = [prepare_sequence(record["frames"]) for record in recordings]
    samples = list(sequences)
    targets = [labels.index(record["label"]) for record in recordings]
    for sequence in sequences:
        for index in (0, FRAMES // 2, FRAMES - 1):
            stationary = np.repeat(sequence[index:index + 1], FRAMES, axis=0)
            stationary[:, -2:] = 0
            samples.append(stationary)
            targets.append(labels.index(NO_MOTION))

    network = build_network(len(labels), sequences[0].shape[1])
    network.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    network.fit(np.asarray(samples), np.asarray(targets), epochs=200, verbose=0)
    return {"recordings": recordings, "labels": labels, "weights": network.get_weights()}


class TensorFlowMotionRecognizer:
    def __init__(self, state, feature_extractor=frame_features):
        self.frames = deque(maxlen=FRAMES)
        self.labels = state["labels"]
        self.feature_extractor = feature_extractor
        self.network = None
        if state["weights"] is not None:
            feature_count = state["weights"][0].shape[0] // FRAMES
            self.network = build_network(len(self.labels), feature_count)
            self.network.set_weights(state["weights"])

    def reset(self):
        self.frames.clear()

    def update(self, landmarks):
        if self.network is None:
            return None
        self.frames.append(self.feature_extractor(landmarks))
        if len(self.frames) < FRAMES:
            return None
        sequence = prepare_sequence(self.frames)
        if not has_motion(sequence):
            return None
        scores = self.network(sequence[np.newaxis, ...], training=False).numpy()[0]
        index = int(np.argmax(scores))
        if self.labels[index] != NO_MOTION and scores[index] >= 0.9:
            self.reset()
            return self.labels[index]
        return None
