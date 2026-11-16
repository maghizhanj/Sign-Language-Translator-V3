REST_LABEL = "__rest__"
STABLE_ENTRY_FRAMES = 6
REST_ENTRY_FRAMES = 2


class AutomaticEntry:
    def __init__(self):
        self.reset()

    def reset(self):
        self.ready = True
        self.last_entry = None
        self.candidate = None
        self.candidate_frames = 0

    def observe(self, label, motion=False):
        if not label:
            self.candidate = None
            self.candidate_frames = 0
            return None

        if motion:
            self.candidate = None
            self.candidate_frames = 0
        elif label == self.candidate:
            self.candidate_frames += 1
        else:
            self.candidate = label
            self.candidate_frames = 1

        required_frames = REST_ENTRY_FRAMES if label == REST_LABEL else STABLE_ENTRY_FRAMES
        if not motion and self.candidate_frames < required_frames:
            return None
        if label == REST_LABEL:
            self.last_entry = None
            return None
        if label == self.last_entry:
            return None
        self.last_entry = label
        return label
