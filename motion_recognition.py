from math import hypot


def _distance(first, second):
    return hypot(first[0] - second[0], first[1] - second[1])


def recognize_j_path(points):
    if len(points) < 5 or sum(
        _distance(points[index - 1], points[index])
        for index in range(1, len(points))
    ) < 0.45:
        return False

    deepest_index = max(range(len(points)), key=lambda index: points[index][1])
    start = points[0]
    deepest = points[deepest_index]
    end = points[-1]

    return (
        len(points) * 0.45 <= deepest_index <= len(points) * 0.95
        and deepest[1] - start[1] >= 0.28
        and abs(deepest[0] - start[0]) <= 0.8
        and abs(end[0] - deepest[0]) >= 0.12
        and end[1] >= deepest[1] - 0.18
    )


def _sample_path(points, fraction):
    distances = [
        _distance(points[index - 1], points[index])
        for index in range(1, len(points))
    ]
    total_distance = sum(distances)
    target_distance = total_distance * fraction
    traversed = 0.0

    for index, segment_distance in enumerate(distances, start=1):
        if traversed + segment_distance >= target_distance:
            segment_fraction = (
                (target_distance - traversed) / segment_distance
                if segment_distance
                else 0
            )
            previous = points[index - 1]
            current = points[index]
            return (
                previous[0] + (current[0] - previous[0]) * segment_fraction,
                previous[1] + (current[1] - previous[1]) * segment_fraction,
            )
        traversed += segment_distance

    return points[-1]


def recognize_z_path(points):
    if len(points) < 5:
        return False

    samples = [_sample_path(points, fraction) for fraction in (0, 1 / 3, 2 / 3, 1)]
    first, diagonal, last = (
        (
            samples[index + 1][0] - samples[index][0],
            samples[index + 1][1] - samples[index][1],
        )
        for index in range(3)
    )

    return (
        abs(first[0]) >= 0.18
        and abs(first[1]) <= abs(first[0]) * 0.7
        and abs(diagonal[0]) >= 0.12
        and abs(diagonal[1]) >= 0.12
        and first[0] * diagonal[0] < 0
        and abs(last[0]) >= 0.18
        and abs(last[1]) <= abs(last[0]) * 0.7
        and first[0] * last[0] > 0
        and max(point[1] for point in points) - min(point[1] for point in points) >= 0.2
    )


class MotionRecognizer:
    def __init__(self):
        self.previous = {}
        self.paths = {"J": None, "Z": None}
        self.quiet_frames = {"J": 0, "Z": 0}
        self.cooldown = 0

    def reset(self):
        self.previous.clear()
        self.paths = {"J": None, "Z": None}
        self.quiet_frames = {"J": 0, "Z": 0}
        self.cooldown = 0

    def update(self, landmarks, disabled=None):
        if not landmarks or len(landmarks) < 21:
            self.reset()
            return None

        wrist = landmarks[0]
        palm_width = _distance(landmarks[5], landmarks[17])
        if palm_width < 1:
            self.reset()
            return None

        tips = {
            "J": (20, 18),
            "Z": (8, 6),
        }
        current = {}

        for letter, (tip_index, pip_index) in tips.items():
            if disabled and letter in disabled:
                self.paths[letter] = None
                self.quiet_frames[letter] = 0
                self.previous.pop(letter, None)
                continue
            tip = landmarks[tip_index]
            pip = landmarks[pip_index]
            if _distance(tip, wrist) <= _distance(pip, wrist) + palm_width * 0.05:
                self.paths[letter] = None
                self.quiet_frames[letter] = 0
                continue
            current[letter] = (
                (tip[0] - wrist[0]) / palm_width,
                (tip[1] - wrist[1]) / palm_width,
            )

        if self.cooldown:
            self.cooldown -= 1
            self.previous = current
            self.paths = {"J": None, "Z": None}
            return None

        for letter, point in current.items():
            previous = self.previous.get(letter)
            path = self.paths[letter]
            movement = _distance(previous, point) if previous is not None else 0

            if path is None:
                if previous is not None and movement >= 0.012:
                    self.paths[letter] = [previous, point]
                    self.quiet_frames[letter] = 0
            elif movement >= 0.006 and len(path) < 100:
                path.append(point)
                self.quiet_frames[letter] = 0
            else:
                self.quiet_frames[letter] += 1

            self.previous[letter] = point

            if self.paths[letter] and self.quiet_frames[letter] >= 7:
                path = self.paths[letter]
                detected = (
                    recognize_j_path(path)
                    if letter == "J"
                    else recognize_z_path(path)
                )
                self.paths[letter] = None
                self.quiet_frames[letter] = 0
                if detected:
                    self.cooldown = 20
                    return letter

        for letter in self.paths:
            if letter not in current:
                self.previous.pop(letter, None)

        return None
